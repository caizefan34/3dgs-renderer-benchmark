import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, os.path.join(repo, "src"))
sys.path.insert(0, repo)
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint
from gsplat import rasterization
print("All imports OK")
