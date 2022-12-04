
from .abstract_dataset import AbstractDataset
from .cvpr2022_df import Cvpr2022DF
# from .faceforensics import FaceForensics
# from .wild_deepfake import WildDeepfake
# from .celeb_df import CelebDF
# from .celeb_df_sbi import CelebDFSBI
# from .dfdc import DFDC


LOADERS = {
    # "FaceForensics": FaceForensics,
    # "WildDeepfake": WildDeepfake,
    # "CelebDF": CelebDF,
    # 'CelebDFSBI': CelebDFSBI,
    # "DFDC": DFDC,
    'Cvpr2022Df': Cvpr2022DF
}


def load_dataset(name="FaceForensics"):
    print(f"Loading dataset: '{name}'...")
    return LOADERS[name]
