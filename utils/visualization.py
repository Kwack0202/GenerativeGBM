from common_imports import *

# 시뮬레이션 결과 시각화 함수 (출력 대신 PNG 파일로 저장)
def plot_simulations(train_close, fake_prices, test_close, simulated_S, num_plot=10, save_path='plot_simulations.png'):
    plt.figure(figsize=(15, 7))
    
    # 학습 기간의 실제 데이터 플롯
    plt.plot(train_close, color='blue', label='Train Real Data', linewidth=1)
    
    # 학습 기간의 가상 데이터 플롯
    plt.plot(fake_prices, color='orange', label='Train Fake Data', linewidth=1)
    
    # 시뮬레이션된 경로 플롯 
    for i in range(num_plot):
        plt.plot(range(len(train_close), len(train_close) + simulated_S.shape[1]), 
                 simulated_S[i], alpha=0.5, linewidth=1, 
                 label='GBM-Data' if i == 0 else "")
    
    # 테스트 기간의 실제 데이터 플롯
    plt.plot(range(len(train_close), len(train_close) + len(test_close)), 
             test_close, color='red', label='Test Real Data', linewidth=1)
    
    plt.title('GenerativeAI-GBM')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    # 지정한 경로에 저장 (디렉토리가 없으면 생성)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close('all')

def plot_confidence_interval(train_close, lower_bound, upper_bound, test_close, simulated_S, num_plot=10, save_path='confidence_interval.png'):
    """
    Train 구간의 실제 가격과 GAN으로 생성된 10개 가상 데이터의 95% 신뢰구간을 시각화합니다.
    
    Args:
        train_close (array-like): 실제 train 구간의 주가 (누적합 적용 후, 실제 가격).
        lower_bound (array-like): 각 시점별 2.5번째 백분위수 값.
        upper_bound (array-like): 각 시점별 97.5번째 백분위수 값.
        save_path (str): 결과 플롯을 저장할 경로.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(train_close, color='blue', label='Train Real Data', linewidth=1)
    plt.fill_between(range(len(train_close)), lower_bound, upper_bound, color='orange', alpha=0.3, label='Confidence Interval')
    
    # 시뮬레이션된 경로 플롯 
    for i in range(num_plot):
        plt.plot(range(len(train_close), len(train_close) + simulated_S.shape[1]), 
                 simulated_S[i], alpha=0.5, linewidth=1, 
                 label='GBM-Data' if i == 0 else "")
    
    # 테스트 기간의 실제 데이터 플롯
    plt.plot(range(len(train_close), len(train_close) + len(test_close)), 
             test_close, color='red', label='Test Real Data', linewidth=1)
    
    plt.title('Train Data with Confidence Interval from GAN-generated Fake Data')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close('all')
    
# 노이즈 분포 시각화 함수 (출력 대신 PNG 파일로 저장하고, K-S test 결과를 txt 파일로 저장)
def visualize_noise_distribution(real_noise, generated_noise, 
                                 hist_save_path='noise_distribution.png', 
                                 qq_save_path='qq_plot.png', 
                                 ks_output_path='ks_test.txt'):
    # 히스토그램 플롯 생성 및 저장
    plt.figure(figsize=(10, 6))
    plt.hist(real_noise, bins=100, alpha=0.5, label='Real Noise', density=True)
    plt.hist(generated_noise, bins=100, alpha=0.5, label='Generated Noise', density=True)
    plt.legend()
    plt.title('Real vs. Generated Noise Distribution')
    plt.xlabel('Noise Value')
    plt.ylabel('Density')
    
    os.makedirs(os.path.dirname(hist_save_path), exist_ok=True)
    plt.savefig(hist_save_path)
    plt.close('all')

    # K-S 테스트 수행
    ks_stat, ks_pvalue = ks_2samp(real_noise, generated_noise)
    
    # Q-Q Plot 생성 및 저장
    plt.figure(figsize=(10, 6))
    sm.qqplot(generated_noise, line='45', fit=True)
    plt.title('Q-Q Plot of Generated Noise')
    
    os.makedirs(os.path.dirname(qq_save_path), exist_ok=True)
    plt.savefig(qq_save_path)
    plt.close('all')
    
    # K-S test 결과를 텍스트 파일로 저장
    os.makedirs(os.path.dirname(ks_output_path), exist_ok=True)
    with open(ks_output_path, 'w') as f:
        f.write(f'K-S Statistic: {ks_stat:.4f}, P-value: {ks_pvalue:.4f}\n')