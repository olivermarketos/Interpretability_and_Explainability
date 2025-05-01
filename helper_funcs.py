from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime

def init_writer(config):

    base_output_dir = Path(config["output_base_dir"])
    base_output_dir.mkdir(exist_ok=True, parents=True)

    log_dir = base_output_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    tensorboard_log_parent_dir = base_output_dir / "logs" 
    tensorboard_log_parent_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique name for this specific run, e.g., using a timestamp
    # or combining it with your model identifier.
   
    # Define the specific log directory for THIS run
    run_name = config["run_name"] 
    current_run_log_dir = tensorboard_log_parent_dir / run_name

    # Initialize the writer to use the unique directory
    writer = SummaryWriter(log_dir=str(current_run_log_dir))

    return writer