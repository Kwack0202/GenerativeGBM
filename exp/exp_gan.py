from common_imports import *
from exp.exp_basic import Exp_Basic
from models.VanillaGAN import Generator, Discriminator
from models.WGAN import WGenerator, WDiscriminator
from models.QuantGAN import QuantGenerator, QuantDiscriminator
from data.data_loader import StockDataset
from utils.sliding_window import get_sliding_window_data
from utils.gmm_preprecessing import *
from utils.visualization import *

class Exp_GAN(Exp_Basic):
    def __init__(self, args):
        super(Exp_GAN, self).__init__(args)

        self.netG = None  
        self.Ticker_log_mean = None
        self.Ticker_max = None
        self.params = None 
    
    def _get_data(self, Ticker_log_df):
        Ticker_log_mean = np.mean(Ticker_log_df)
        Ticker_log_norm = Ticker_log_df - Ticker_log_mean
        params = igmm(Ticker_log_norm)
        Ticker_processed = W_delta((Ticker_log_norm - params[0]) / params[1], params[2])
        Ticker_max = np.max(np.abs(Ticker_processed))
        Ticker_processed /= Ticker_max
        
        return Ticker_log_mean, params, Ticker_max, Ticker_processed
    
    def _build_generator(self):
        
        if self.args.model_name == 'VanillaGAN':   
            netG = Generator(self.args.noise_input_size, self.args.noise_output_size).to(self.device)
        
        elif self.args.model_name == 'WGAN':   
            netG = WGenerator(self.args.noise_input_size, self.args.noise_output_size).to(self.device)
            
        elif self.args.model_name == 'QuantGAN':   
            netG = QuantGenerator(self.args.noise_input_size, self.args.noise_output_size).to(self.device)
           
        return netG

    def _build_discriminator(self):
        
        if self.args.model_name == 'VanillaGAN':
            netD = Discriminator(self.args.noise_output_size).to(self.device)
        
        elif self.args.model_name == 'WGAN':
            netD = WDiscriminator(self.args.noise_output_size).to(self.device)
        
        elif self.args.model_name == 'QuantGAN':
            netD = QuantDiscriminator(self.args.noise_output_size, self.args.noise_output_size).to(self.device)
        
        return netD
            
    def _select_optimizer(self, netG, netD):
        
        if self.args.model_optimizer == 'Adam':
            optG = optim.Adam(netG.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))
            optD = optim.Adam(netD.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))
            
        elif self.args.model_optimizer == 'RMSprop':
            optG = optim.RMSprop(netG.parameters(), lr=self.args.learning_rate)
            optD = optim.RMSprop(netD.parameters(), lr=self.args.learning_rate)
            
        else:
            raise ValueError("Unknown optimizer specified")
        return optG, optD
    
    def _save_settings(self):
        """
        Save the values of all parser arguments passed in run.py to the settings.txt file.
        """
        settings_folder = f'./outputs/{self.args.model_type}/{self.args.model_name}/'
        os.makedirs(settings_folder, exist_ok=True)
        settings_path = os.path.join(settings_folder, 'settings.txt')
        with open(settings_path, 'w') as f:
            for key, value in vars(self.args).items():
                f.write(f"{key}: {value}\n")
    
                            
    def train_gan(self):
        self._save_settings()
        
        root_path = self.args.exp_root_path
        stock_tickers = [f for f in os.listdir(root_path) if f.endswith('.csv')]
        
        for ticker in stock_tickers:
            print(f'\nGeneraticeAI-GBM start ~~ Stock code : {ticker[:-4]} ')
            stock_data = pd.read_csv(os.path.join(root_path, ticker))
            
            windowed_data = get_sliding_window_data(
                stock_data,
                date_col='index',
                test_start_year=self.args.test_start_year,
                total_test_months=self.args.total_test_months,
                sliding_test_months=self.args.sliding_test_months,
                train_months=self.args.train_months
            )
            
            # results save root (model pth file, plot, txt etc.)
            output_root = f'./outputs/{self.args.model_type}/{self.args.model_name}/{ticker[:-4]}/Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/'
                
            for num_window, (train_df, test_df, window_info) in enumerate(windowed_data):
                print(f"--- Sliding Window {num_window+1} ---")
                print(f"  Train period: {window_info['train_start']} ~ {window_info['train_end']} | Rows: {window_info['train_rows']}")
                print(f"  Test period: {window_info['test_start']} ~ {window_info['test_end']} | Rows: {window_info['test_rows']}")

                # ====================================
                # step 1. Data preprecessing (GMM)
                Ticker_log_train = np.log(train_df['Adj Close'] / train_df['Adj Close'].shift(1))[1:].values
                Ticker_log_mean, params, Ticker_max, Ticker_processed = self._get_data(Ticker_log_train)
                
                self.Ticker_log_mean = Ticker_log_mean
                self.params = params
                self.Ticker_max = Ticker_max
                
                # ====================================
                # step 2. Model define
                netG = self._build_generator()
                netD = self._build_discriminator()
                optG, optD = self._select_optimizer(netG, netD)
                
                # ====================================
                # step 3. Dataset load
                dataset = StockDataset(Ticker_processed, self.args.seq_len)
                dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=self.args.batch_size, 
                                         shuffle=True, 
                                         num_workers=self.args.num_workers)
                
                # ====================================
                # step 4. Model train                
                progress = tqdm(range(self.args.num_epochs))
                
                best_loss = float('inf')
                early_stop_counter = 0
                best_epoch = 0
                best_netG = None
                best_netD = None
                
                for epoch in progress:
                    for i, data in enumerate(dataloader, 0):
                        
                        netD.zero_grad()
                        real = data.to(self.device)
                        batch_size, seq_len = real.size(0), real.size(1)
                        noise = torch.randn(batch_size, seq_len, self.args.noise_input_size, device=self.device)
                        fake = netG(noise).detach()

                        # Discriminator                        
                        if self.args.model_name == 'VanillaGAN':
                            lossD = -(torch.mean(torch.log(netD(real))) + torch.mean(torch.log(1. - netD(fake))))
                            lossD.backward()
                            optD.step()
                        
                        elif self.args.model_name in ['WGAN', 'QuantGAN']:
                            lossD = -torch.mean(netD(real)) + torch.mean(netD(fake))
                            lossD.backward()
                            optD.step()
                            
                            for p in netD.parameters():
                                p.data.clamp_(-0.01, 0.01)
                                            
                        # Generator
                        if i % 5 == 0:
                            netG.zero_grad()
                            if self.args.model_name == 'VanillaGAN':
                                lossG = torch.mean(torch.log(1. - netD(netG(noise))))
                            elif self.args.model_name in ['WGAN', 'QuantGAN']:
                                lossG = -torch.mean(netD(netG(noise)))
                            lossG.backward()
                            optG.step()
                    
                    # ==============================================
                    # Metric 1.Confidence coverage 
                    P0 = train_df['Adj Close'].iloc[0]
                    real_cumsum = Ticker_log_train.cumsum()
                    real_prices = P0 * np.exp(real_cumsum)
                    
                    fake_data = self.generate_fakes(Ticker_log_train, self.args.noise_input_size, cumsum=True, n=self.args.fake_sample, generator=netG)
                    clipped_values = np.clip(fake_data.values, real_cumsum.min(), real_cumsum.max()) # Preventing overflows 
                    fake_array = P0 * np.exp(clipped_values)
                    
                    # confidence interval
                    conf_level = (1-self.args.confidence) / 2 * 100
                    lower_bound = np.percentile(fake_array, conf_level, axis=1)
                    upper_bound = np.percentile(fake_array, 100 - conf_level, axis=1)
                    
                    coverage = np.mean((real_prices >= lower_bound) & (real_prices <= upper_bound))
                    
                    # ==============================================
                    # Metric 2.K-S Test 
                    generated_noise = self.extract_learned_noise(self.args.num_noise_samples, seq_length=len(Ticker_log_train), nz=self.args.noise_input_size, generator = netG)
                    real_noise = Ticker_processed.flatten()
                    ks_stat, ks_pvalue = ks_2samp(real_noise, generated_noise)
                        
                    msg = f'Loss_D: {lossD.item():.8f} | Loss_G: {lossG.item():.8f} | KS statistic: {ks_stat:.6f} (p-value: {ks_pvalue:.4f}) | Coverage: {coverage:.4f}' 
                    
                    progress.set_description(msg)
                    
                    # ==============================================
                    # Monitoring & EarlyStop
                    if (epoch + 1) % self.args.check_interval == 0 and (epoch + 1) >= self.args.min_epochs:
                        # KS, p-value, Coverage 조건이 모두 만족하면 loss 개선 여부를 체크
                        if ks_stat < self.args.ks_threshold and ks_pvalue > self.args.pvalue_threshold and coverage > self.args.coverage_threshold:
                            # lossG가 지정한 tolerance 이상 개선되었으면 best_loss 갱신
                            if lossG.item() < best_loss - self.args.loss_tolerance:
                                best_loss = lossG.item()
                                early_stop_counter = 0
                                best_epoch = epoch + 1
                                best_netG = copy.deepcopy(netG)
                                best_netD = copy.deepcopy(netD)
                            else:
                                early_stop_counter += 1
                            # patience 이상 개선이 없으면 얼리 스탑
                            if early_stop_counter >= self.args.early_stop_patience:
                                print(f"[Early Stopping] Epoch {epoch+1}: Loss improvement has not occurred for {early_stop_counter} consecutive times."
                                      f"KS statistic = {ks_stat:.6f}, KS p-value = {ks_pvalue:.4f}, Coverage = {coverage:.4f}")
                                break
                    # stop epoch loop
                    if early_stop_counter >= self.args.early_stop_patience:
                        break
                    
                # Checkpoint
                sliding_folder = os.path.join(output_root, f'Sliding_Window_{num_window+1}_')
                os.makedirs(sliding_folder, exist_ok=True)
                if best_netG is not None:
                    netG = best_netG
                    netD = best_netD
                    final_epoch = best_epoch
                else:
                    final_epoch = epoch + 1
                
                torch.save(netG, os.path.join(sliding_folder, f'netG_epoch_{final_epoch}.pth'))
                torch.save(netD, os.path.join(sliding_folder, f'netD_epoch_{final_epoch}.pth'))
                
                # ====================================
                # step 5. Simulation
                self.netG = torch.load(os.path.join(sliding_folder, f'netG_epoch_{final_epoch}.pth'))
                self.netG.eval()

                                
                '''
                Generate Fake Data using GAN (Train period)
                '''
                real_cumsum = Ticker_log_train.cumsum()
                fakes_cumsum = self.generate_fakes(Ticker_log_train, self.args.noise_input_size, cumsum=True).flatten()
                
                P0 = train_df['Adj Close'].iloc[0]
                real_prices = P0 * np.exp(real_cumsum)
                fake_prices = P0 * np.exp(fakes_cumsum)
                
                '''
                Generate Confidence coverage using GAN (Train period)
                '''
                fake_data = self.generate_fakes(Ticker_log_train, self.args.noise_input_size, cumsum=True, n=self.args.fake_sample)
                fake_array = P0 * np.exp(fake_data.values)
                
                conf_level = (1-self.args.confidence) / 2 * 100
                lower_bound = np.percentile(fake_array, conf_level, axis=1)
                upper_bound = np.percentile(fake_array, 100 - conf_level, axis=1)                 
                        
                '''
                Generate Fake Data using GBM (Test period)
                '''                
                test_length = len(test_df) - 1  # Test length excluding first day price
                T = test_length / 252  # Convert to year (based on trading day)
                dt = 1/252 # Days

                # GBM MonteCarlo Simulation
                test_close = test_df['Adj Close'].values
                S0_price = train_df['Adj Close'].iloc[-1]
                
                
                simulated_S = self.simulate_gbm_multiple(
                    S0=S0_price,
                    mu=np.mean(Ticker_log_train)*500,
                    sigma=np.std(Ticker_log_train)*20,
                    T=T,
                    dt=dt,
                    generator=self.netG,
                    nz=self.args.noise_input_size,
                    num_simulations=self.args.num_simulations,
                )
                plot_simulations(real_prices, fake_prices, test_close, simulated_S, num_plot=self.args.num_simulations,
                                 save_path=os.path.join(output_root, f'Sliding_Window_{num_window+1}_', 'simulation_result.png'))
                
                plot_confidence_interval(real_prices, lower_bound, upper_bound, test_close, simulated_S, num_plot=10, 
                                         save_path=os.path.join(output_root, f'Sliding_Window_{num_window+1}_', f'confidence_results.png'))
                    
                # 학습된 노이즈 분포 추출 및 시각화
                learned_noise = self.extract_learned_noise(self.args.num_noise_samples, seq_length=len(Ticker_log_train), nz=self.args.noise_input_size, device=self.device)
                real_noise = Ticker_processed.flatten()
                
                ks_stat, ks_pvalue = calculate_ks_test(real_noise, learned_noise)
                save_metric_txt(ks_stat, ks_pvalue, coverage, ks_output_path=os.path.join(output_root, f'Sliding_Window_{num_window+1}_', 'save_metric.txt'))
                
                # 분포 시각화
                visualize_noise_distribution(real_noise, learned_noise, 
                                            hist_save_path=os.path.join(output_root, f'Sliding_Window_{num_window+1}_','noise_distribution.png'),
                                            qq_save_path=os.path.join(output_root, f'Sliding_Window_{num_window+1}_','qq_plot.png'))

        
                torch.cuda.empty_cache()    
    
    def generate_fakes(self, Ticker_log_train, nz, cumsum=True, n=1, generator=None):
        """
        Using generators to generate virtual data(log return) and inverse scale(Cumsumed real price)
        """
        if generator is None:
            generator = self.netG
                    
        fakes = []
        for i in range(n):
            noise = torch.randn(1, len(Ticker_log_train), nz, device=self.device)
            fake = generator(noise).detach().cpu().reshape(len(Ticker_log_train)).numpy()
            Ticker_fake = inverse(fake * self.Ticker_max, self.params) + self.Ticker_log_mean
            fakes.append(Ticker_fake)
            
        if n > 1:
            if not cumsum:
                return pd.DataFrame(fakes).T
            fakes_df = pd.DataFrame(fakes).T.cumsum()
            return fakes_df
        elif not cumsum:
            return Ticker_fake
        return Ticker_fake.cumsum()
    
    def extract_learned_noise(self, num_sequences, seq_length, nz, generator=None, device=None):
        """
        Extract noise samples using a generator.
        """
        if device is None:
            device = self.device
        if generator is None:
            generator = self.netG
        generator.eval()
        noise_samples = []
        with torch.no_grad():
            for _ in range(num_sequences):
                noise = torch.randn(1, seq_length, nz, device=device)
                fake = generator(noise).cpu().numpy().flatten()
                noise_samples.extend(fake)
        return np.array(noise_samples)
    

    def simulate_gbm_multiple(self, S0, mu, sigma, T, dt, nz, num_simulations, generator=None, device=None):
        """
        Simulate multiple GBM paths using the extracted noise.
        """
        if device is None:
            device = self.device
        if generator is None:
            generator = self.netG
        N = int(T / dt)
        S = np.zeros((num_simulations, N))
        S[:, 0] = S0
        learned_noises = self.extract_learned_noise(num_sequences=num_simulations, seq_length=N-1, nz=nz, generator=generator, device=device)
        learned_noises = learned_noises.reshape(num_simulations, N-1)
        for t in range(1, N):
            S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma * learned_noises[:, t-1] * np.sqrt(dt))
        return S
    
    def simulate(self):
        stock_files = [f for f in os.listdir(self.args.exp_root_path) if f.endswith('.csv')]
        
        for stock_file in stock_files:
            ticker_name = stock_file[:-4]  # 예: "AAPL.csv" → "AAPL"
            print(f"\n[INFO] Simulating for ticker: {ticker_name}")
            
            # stock data load
            stock_data = pd.read_csv(os.path.join(self.args.exp_root_path, stock_file))
            
            sliding_windows = get_sliding_window_data(
                stock_data,
                date_col='index',
                test_start_year=self.args.test_start_year,
                total_test_months=self.args.total_test_months,
                sliding_test_months=self.args.sliding_test_months,
                train_months=self.args.train_months
            )
            
            if len(sliding_windows) == 0:
                print(f"[WARNING] 슬라이딩 윈도우 데이터가 없습니다. {ticker_name}는 건너뜁니다.")
                continue
            
            # model pth file root (ex: ./outputs/GAN/VanillaGAN/AAPL/Train_36_Test_12/)
            base_output_folder = os.path.join('./outputs', self.args.model_type, self.args.model_name, ticker_name,
                                              f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}")
            if not os.path.exists(base_output_folder):
                print(f"[WARNING] 출력 폴더 {base_output_folder} 가 존재하지 않습니다. {ticker_name}는 건너뜁니다.")
                continue
            
            # Simulation
            for window_idx, (train_df, test_df, window_info) in enumerate(sliding_windows):
                sliding_folder = os.path.join(base_output_folder, f"Sliding_Window_{window_idx+1}")
                netG_file = os.path.join(sliding_folder, f"netG_epoch_{self.args.num_epochs}.pth")
                
                if not os.path.exists(netG_file):
                    print(f"[WARNING] {netG_file} 가 존재하지 않습니다. {ticker_name}의 window {window_idx+1} 건너뜁니다.")
                    continue
                
                print(f"[INFO] {ticker_name} - Window {window_idx+1}: 모델 로드 중: {netG_file}")
                self.netG = torch.load(netG_file, map_location=self.device)
                self.netG.eval()
                
                # 테스트 구간 관련 파라미터 계산
                test_length = len(test_df) - 1  # 첫 행은 S₀로 사용
                T = test_length / 252    # 거래일 기준 연 단위
                dt = 1 / 252
                S0_price = train_df['Adj Close'].iloc[-1]  # 학습 구간 마지막 종가를 초기 가격으로 사용
                
                # 학습 구간의 로그수익률을 기반으로 drift와 volatility 계산
                Ticker_log_train = np.log(train_df['Adj Close'] / train_df['Adj Close'].shift(1))[1:].values
                mu = np.mean(Ticker_log_train) * 500
                sigma = np.std(Ticker_log_train) * 20
                
                print(f"[INFO] Window {window_idx+1}: S0 = {S0_price}, mu = {mu:.4f}, sigma = {sigma:.4f}, T = {T:.4f}")
                
                # GBM 시뮬레이션 (이미 simulate_gbm_multiple에 netG를 사용하여 노이즈 추출)
                simulated_paths = self.simulate_gbm_multiple(
                    S0=S0_price,
                    mu=mu,
                    sigma=sigma,
                    T=T,
                    dt=dt,
                    nz=self.args.noise_input_size,
                    num_simulations=self.args.num_simulations,
                )
                
                # 결과 DataFrame 생성: 행은 시점, 열은 각 시뮬레이션 경로
                df_simulated = pd.DataFrame(
                    simulated_paths.T,
                    columns=[f"simulation_{i+1}" for i in range(self.args.num_simulations)]
                )
                
                # CSV 파일로 저장 (예: simulation_result.csv)
                simulation_file = os.path.join(sliding_folder, "simulated_paths.csv")
                df_simulated.to_csv(simulation_file, index=False)
                print(f"[INFO] Window {window_idx+1}: 시뮬레이션 결과 저장 완료 → {simulation_file}")