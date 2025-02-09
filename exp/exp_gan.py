from common_imports import *
from exp.exp_basic import Exp_Basic
from models.VanillaGAN import Generator, Discriminator
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
    
    def _get_data(self, Ticker_log_df): # 데이터의 평균과 정규화
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
           
        return netG

    def _build_discriminator(self):
        
        if self.args.model_name == 'VanillaGAN':
            netD = Discriminator(self.args.noise_output_size).to(self.device)
        
        return netD
            
    def _select_optimizer(self, netG, netD):
        
        if self.args.model_optimizer == 'Adam':
            optG = optim.Adam(netG.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))
            optD = optim.Adam(netD.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))
            
        return optG, optD
    
    def train_gan(self):
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
                Ticker_log_train = np.log(train_df['Close'] / train_df['Close'].shift(1))[1:].values
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
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
                
                # ====================================
                # step 4. Model train
                progress = tqdm(range(self.args.num_epochs))
                for epoch in progress:
                    for i, data in enumerate(dataloader, 0):
                        
                        netD.zero_grad()
                        real = data.to(self.device)
                        batch_size, seq_len = real.size(0), real.size(1)
                        noise = torch.randn(batch_size, seq_len, self.args.noise_input_size, device=self.device)
                        fake = netG(noise).detach()

                        # Discriminator
                        lossD_real = torch.mean(torch.log(netD(real)))
                        lossD_fake = torch.mean(torch.log(1. - netD(fake)))
                        lossD = -(lossD_real + lossD_fake)
                        lossD.backward()
                        optD.step()

                        # Generator
                        if i % 5 == 0:
                            netG.zero_grad()
                            lossG = torch.mean(torch.log(1. - netD(netG(noise))))
                            lossG.backward()
                            optG.step()

                    progress.set_description(f'Loss_D: {lossD.item():.8f} Loss_G: {lossG.item():.8f}')
                    
                    if not os.path.exists(output_root + f'Sliding_Window_{num_window+1}/'):
                        os.makedirs(output_root + f'Sliding_Window_{num_window+1}/')
                            
                # Checkpoint
                torch.save(netG, output_root + f'Sliding_Window_{num_window+1}/netG_epoch_{self.args.num_epochs}.pth')
                torch.save(netD, output_root + f'Sliding_Window_{num_window+1}/netD_epoch_{self.args.num_epochs}.pth')

                # ====================================
                # step 5. Simulation
                self.netG = torch.load(output_root + f'Sliding_Window_{num_window+1}/netG_epoch_{self.args.num_epochs}.pth')
                self.netG.eval()
                                
                '''
                Generate Fake Data using GAN (Train period)
                '''
                real_cumsum = Ticker_log_train.cumsum()
                fakes_cumsum = self.generate_fakes(Ticker_log_train, self.args.noise_input_size, cumsum=True).flatten()
                
                P0 = train_df['Close'].iloc[0]
                real_prices = P0 * np.exp(real_cumsum)
                fake_prices = P0 * np.exp(fakes_cumsum)
                
                '''
                Generate Fake Data using GBM (Test period)
                '''
                Ticker_log_test = np.log(test_df['Close'] / test_df['Close'].shift(1))[1:].values

                # 테스트 데이터의 로그 수익률 정규화
                Ticker_log_test_norm = Ticker_log_test - Ticker_log_mean
                Ticker_processed_test = W_delta((Ticker_log_test_norm - params[0]) / params[1], params[2])
                Ticker_processed_test /= Ticker_max
                
                test_length = len(test_df) - 1  # 첫 날의 S0를 제외한 길이
                T = test_length / 252  # 연 단위로 변환 (거래일 기준)
                dt = 1/252  # 일 단위

                # GBM MonteCarlo Simulation
                test_close = test_df['Close'].values
                S0_price = train_df['Close'].iloc[-1]
                
                
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
                                 save_path=os.path.join(output_root, f'Sliding_Window_{num_window+1}', 'simulation_result.png'))
                
                # 학습된 노이즈 분포 추출 및 시각화
                learned_noise = self.extract_learned_noise(self.args.num_noise_samples, seq_length=len(Ticker_log_train), nz=self.args.noise_input_size, device=self.device)
                
                # 실제 노이즈는 학습 데이터의 정규화된 노이즈
                real_noise = Ticker_processed.flatten()

                # 분포 시각화
                visualize_noise_distribution(real_noise, learned_noise,
                                             hist_save_path=os.path.join(output_root, f'Sliding_Window_{num_window+1}', 'noise_distribution.png'),
                                             qq_save_path=os.path.join(output_root, f'Sliding_Window_{num_window+1}', 'qq_plot.png'),
                                             ks_output_path=os.path.join(output_root, f'Sliding_Window_{num_window+1}', 'ks_test.txt'))
        
            torch.cuda.empty_cache()    
    
    def generate_fakes(self, Ticker_log_train, nz, cumsum=True, n=1, generator=None):
        """
        학습된 생성기를 사용해 가상 로그수익률 데이터를 생성하고,
        역변환(inverse transformation)을 수행하여 실제 로그수익률로 복원한 후,
        누적합(cumsum) 형태로 반환하는 함수.
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
        학습된 생성기를 통해 노이즈 샘플을 추출합니다.
        
        인자 generator가 None이면, self.netG를 사용하며,
        device가 None이면 self.device를 사용합니다.
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
        학습된 생성기를 통해 추출한 노이즈를 사용하여
        다수의 GBM 경로를 시뮬레이션합니다.
        
        인자 generator가 None이면, self.netG를 사용하며,
        device가 None이면 self.device를 사용합니다.
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
        """
        학습된 모든 종목의 슬라이딩 윈도우에 대해, 
        각 윈도우의 test 구간에 대해 GBM 기반 가상 경로를 생성하고,
        결과를 각 슬라이딩 윈도우 폴더에 CSV 파일로 저장합니다.
        """
        # self.args.exp_root_path 에는 원본 주식 데이터 CSV 파일들이 위치한다고 가정합니다.
        stock_files = [f for f in os.listdir(self.args.exp_root_path) if f.endswith('.csv')]
        
        for stock_file in stock_files:
            ticker_name = stock_file[:-4]  # 예: "AAPL.csv" → "AAPL"
            print(f"\n[INFO] Simulating for ticker: {ticker_name}")
            
            # 주식 데이터 로드
            stock_data = pd.read_csv(os.path.join(self.args.exp_root_path, stock_file))
            
            # 슬라이딩 윈도우 데이터 생성 (학습 시와 동일한 파라미터 사용)
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
            
            # 학습 시 저장된 출력 폴더 경로 (예: ./outputs/GAN/VanillaGAN/AAPL/Train_36_Test_12/)
            base_output_folder = os.path.join('./outputs', self.args.model_type, self.args.model_name, ticker_name,
                                              f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}")
            if not os.path.exists(base_output_folder):
                print(f"[WARNING] 출력 폴더 {base_output_folder} 가 존재하지 않습니다. {ticker_name}는 건너뜁니다.")
                continue
            
            # 각 슬라이딩 윈도우에 대해 시뮬레이션 실행
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
                S0_price = train_df['Close'].iloc[-1]  # 학습 구간 마지막 종가를 초기 가격으로 사용
                
                # 학습 구간의 로그수익률을 기반으로 drift와 volatility 계산
                Ticker_log_train = np.log(train_df['Close'] / train_df['Close'].shift(1))[1:].values
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
                print(f"    [INFO] Window {window_idx+1}: 시뮬레이션 결과 저장 완료 → {simulation_file}")