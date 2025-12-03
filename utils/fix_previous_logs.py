import os
import glob
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

def fix_event_file(file_path, dry_run=False):
    print(f"Processing {file_path}...")
    # Load only scalars for efficiency, as that's what we want to fix
    ea = EventAccumulator(file_path, size_guidance={'scalars': 0})
    ea.Reload()
    
    scalars = ea.Tags()['scalars']
    if not scalars:
        print("  No scalars found.")
        return

    # Find max step in this file
    max_step = 0
    has_neg_one = False
    for tag in scalars:
        for event in ea.Scalars(tag):
            if event.step == -1:
                has_neg_one = True
            elif event.step > max_step:
                max_step = event.step
    
    if not has_neg_one:
        print("  No step -1 found. Skipping.")
        return

    print(f"  Found step -1. Will replace with max_step={max_step}.")
    
    if dry_run:
        return

    # Create new writer in a 'fixed' subdirectory to avoid overwriting/conflict
    output_dir = os.path.dirname(file_path)
    fixed_log_dir = os.path.join(output_dir, "fixed")
    
    # Use a specific filename to indicate it's fixed, but SummaryWriter controls the prefix
    writer = SummaryWriter(log_dir=fixed_log_dir)
    
    count = 0
    for tag in scalars:
        for event in ea.Scalars(tag):
            step = event.step
            if step == -1:
                step = max_step
            
            # Write the scalar
            writer.add_scalar(tag, event.value, step, walltime=event.wall_time)
            count += 1
            
    writer.close()
    print(f"  Wrote {count} scalars to {fixed_log_dir}")
    print(f"  Please verify the logs in '{fixed_log_dir}' using TensorBoard.")
    print(f"  If correct, you can replace the original files with these.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix TensorBoard logs with step -1")
    parser.add_argument("--root", type=str, default="output", help="Root directory to search for event files")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, do not write")
    args = parser.parse_args()

    print(f"Scanning {args.root} for event files...")
    # Recursive search for event files
    files = glob.glob(os.path.join(args.root, "**", "events.out.tfevents*"), recursive=True)
    
    found = 0
    for f in files:
        if "fixed" in f: continue # Skip already fixed files
        if os.path.isdir(f): continue
        fix_event_file(f, args.dry_run)
        found += 1
        
    if found == 0:
        print("No event files found.")
