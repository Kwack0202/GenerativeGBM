from common_imports import *
from data.data_download import download_stock_data

from exp.exp_gan import Exp_GAN

fix_seed = 42
random.seed(fix_seed)
np.random.seed(fix_seed)

## ==================================================
parser = argparse.ArgumentParser(description="Stock generative GBM")


## ==================================================
## basic config
## ==================================================
parser.add_argument("--task_name", type=str, required=True, default="pretrain", help="task name [options : data_download]")

## ==================================================
## data download
## ==================================================
parser.add_argument('--output_dir', type=str, default='./stock_data/Nasdaq30/', help='origin data directory')
parser.add_argument(
    '--tickers',
    type=str,
    nargs='+',
    default=[
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
        "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
        "WMT", "UNH", "V", "XOM", "MA", 
        "PG", "COST", "JNJ", "ORCL", "HD", 
        "BAC", "KO", "NFLX", "MRK",  "CVX", 
        "CRM", "ADBE", "AMD", "PEP", "TMO"
    ],
    help='List of stock tickers'
    )
parser.add_argument('--start_day', type=str, default='2010-06-01', help='Start date (format : YYYY-MM-DD)')
parser.add_argument('--end_day', type=str, default='2025-01-10', help='End date (format : YYYY-MM-DD)')

## ==================================================
## GenerativeAI modeling
## ==================================================
'''
experiment data root & model define
'''
parser.add_argument('--exp_root_path', type=str, default='./stock_data/Nasdaq30/', help='Data root for experiment')
parser.add_argument('--model_type', type=str, default='GAN', help='generativeAI model type')
parser.add_argument('--model_name', type=str, default='VanillaGAN', help='generativeAI model name')

'''
hyperparameter(sliding window)
'''
parser.add_argument('--test_start_year', type=int, default=2022, help='test_start_year')
parser.add_argument('--total_test_months', type=int, default=36, help='total_test_months')
parser.add_argument('--sliding_test_months', type=int, default=12, help='sliding_test_months')
parser.add_argument('--train_months', type=int, default=36, help='train_months')

'''
hyperparameter(modeling)
'''
parser.add_argument('--noise_input_size', type=int, default=3, help='The noise vector dimension of the generator(input)')
parser.add_argument('--noise_output_size', type=int, default=1, help='The noise vector dimension of the generator(output)')
parser.add_argument('--model_optimizer', type=str, default='Adam', help='model_optimizer')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning_rate')
parser.add_argument('--seq_len', type=int, default=127, help='sequence length')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
parser.add_argument('--num_epochs', type=int, default=2000, help='num_epochs')

'''
hyperparameter(monitoring)
'''
parser.add_argument('--min_epochs', type=int, default=500, help='min_epochs')
parser.add_argument('--check_interval', type=int, default=10, help='check_interval')
parser.add_argument('--ks_threshold', type=float, default=0.05, help='ks_threshold')
parser.add_argument('--pvalue_threshold', type=float, default=0.05, help='pvalue_threshold')
parser.add_argument('--fake_sample', type=int, default=10, help='confidence')
parser.add_argument('--confidence', type=float, default=0.8, help='confidence')
parser.add_argument('--coverage_threshold', type=float, default=0.9, help='coverage_threshold')

'''
hyperparameter(simulate)
'''
parser.add_argument('--num_simulations', type=int, default=10, help='num_simulations')
parser.add_argument('--num_noise_samples', type=int, default=1, help='num_noise_samples')

## ==================================================
'''
Parsing the arguments here
'''
args = parser.parse_args()

if args.task_name == "data_download":
    print("Start downloading the original data")
    
    download_stock_data(
        args.output_dir, 
        args.tickers, 
        args.start_day, 
        args.end_day
        )
    
     
elif args.task_name == 'train':
    print("Start train the generativeAI")
    
    if args.model_type == 'GAN':
        setting = """
        task_name: {}
        exp_root_path: {}
        model_type: {}
        model_name: {}
        test_start_year: {}
        total_test_months: {}
        sliding_test_months: {}
        train_months: {}
        noise_input_size: {}
        noise_output_size: {}
        model_optimizer: {}
        learning_rate: {}
        seq_len: {}
        batch_size: {}
        num_workers: {}
        num_epochs: {}
        min_epochs: {}
        check_interval: {}
        ks_threshold: {}
        pvalue_threshold: {}
        fake_sample: {}
        confidence: {}
        coverage_threshold: {}
        num_simulations:{}
        num_noise_samples:{}""".format(
            args.task_name,
            args.exp_root_path,
            args.model_type,
            args.model_name,
            args.test_start_year,
            args.total_test_months,
            args.sliding_test_months,
            args.train_months,
            args.noise_input_size,
            args.noise_output_size,
            args.model_optimizer,
            args.learning_rate,
            args.seq_len,
            args.batch_size,
            args.num_workers,
            args.num_epochs,
            args.min_epochs,
            args.check_interval,
            args.ks_threshold,
            args.pvalue_threshold,
            args.fake_sample,
            args.confidence,
            args.coverage_threshold,
            args.num_simulations,
            args.num_noise_samples,
        )
        Exp = Exp_GAN
        
        exp = Exp(args)
        print("start training : {}".format(setting))
        
        exp.train_gan()

elif args.task_name == 'simulate':
    print("Start simulation for all sliding windows using GBM for test data")
    
    exp = Exp_GAN(args)
    exp.simulate()