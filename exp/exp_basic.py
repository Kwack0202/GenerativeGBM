from common_imports import *

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        # 모델 종류에 따라 모델을 생성
        self.models = self._build_models()  # 반환값은 딕셔너리 형식
        
    def _acquire_device(self):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        return device
    
    def _build_models(self):
        """
        args.model_type 에 따라 모델 생성:
         - GAN 계열일 경우 : Generator와 Discriminator를 생성하여 딕셔너리에 담아 반환
         - Diffusion 계열일 경우 : 단일 모델을 생성하여 딕셔너리에 담아 반환
        """
        model_type = self.args.model_type
        if model_type == 'GAN':
            generator = self._build_generator()
            discriminator = self._build_discriminator()
            generator = generator.to(self.device)
            discriminator = discriminator.to(self.device)
            
            return {"generator": generator, "discriminator": discriminator}
        
        elif model_type == 'Diffusion':
            model = self._build_diffusion_model()
            model = model.to(self.device)
            return {"Diffusion model": model}
        else:
            raise ValueError(f"Unsupported model_type: {self.args.model_type}")

    # ======================================================
    # Define models
    def _build_generator(self):
        raise NotImplementedError("GAN: Generator is not defined")

    def _build_discriminator(self):
        raise NotImplementedError("GAN: Discriminator is not defined")

    def _build_diffusion_model(self):
        raise NotImplementedError("Diffusion is not defined")

    # ======================================================
    # Data loading, training, testing, etc.
    def _get_data(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError