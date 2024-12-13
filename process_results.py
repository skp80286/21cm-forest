import re
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def parse_log_line(line):
    """Parse a single log line and extract relevant information"""
    try:
        # Initialize default values
        result = {
            'score': 'NA',
            'r2_1': 'NA',
            'r2_2': 'NA',
            'epochs': 'NA',
            'kernel_size': 'NA',
            'points': 'NA',
            'model_param1': 'NA',
            'model_param2': 'NA',
            'telescope': 'NA',
            't_int': 'NA',
            'label': 'NA',
            'ps_bins_to_use': 'NA',            
            'ps_bins_to_make': 'NA',
        }
        
        # Extract values if they exist
        if score_match := re.search(r'score=([\d.]+)', line):
            result['score'] = float(score_match.group(1).rstrip('.'))
            
        if r2_match := re.search(r'r2=\[([\d.]+),\s*([\d.]+)\]', line):
            result['r2_1'] = float(r2_match.group(1).rstrip('.'))
            result['r2_2'] = float(r2_match.group(2).rstrip('.'))
            
        if params_match := re.search(r'Parameters:\s*epochs:\s*(\d+),\s*kernel_sizes:\s*\[(\d+)\],\s*points:\s*(\d+),', line):
            result['epochs'] = int(params_match.group(1).rstrip('.'))
            result['kernel_size'] = int(params_match.group(2).rstrip('.'))
            result['points'] = int(params_match.group(3).rstrip('.'))
            
        if telescope_match := re.search(r'--telescope\s+(\w+)', line):
            result['telescope'] = telescope_match.group(1)
            
        if t_int_match := re.search(r'--t_int\s+(\d+)', line):
            result['t_int'] = int(t_int_match.group(1).rstrip('.'))
            
        if label_match := re.search(r'label=([^\.]+)', line):
            result['label'] = label_match.group(1).strip()

        if ps_bins_to_make_match := re.search(r'ps_bins?_to_use=(\d+)', line):
            result['ps_bins_to_use'] = ps_bins_to_make_match.group(1)
            
        if ps_bins_to_use_match := re.search(r'ps_bins?_to_make=(\d+)', line):
            result['ps_bins_to_make'] = ps_bins_to_use_match.group(1)
            
        if model_param1_match := re.search(r'model_param1=(\d+)', line):
            result['model_param1'] = int(model_param1_match.group(1).rstrip('.'))
            
        if model_param2_match := re.search(r'model_param2=(\d+)', line):
            #eprint(f"Found model_param2 {model_param2_match.group(1)} {model_param2_match.group(1).rstrip('.')}")
            result['model_param2'] = int(model_param2_match.group(1).rstrip('.'))
            
        return result
        
    except (ValueError) as e:
        eprint(f"Error parsing line: {e}")
        return None

def process_file(filename):
    """Process the log file and output CSV format"""
    # Print CSV header
    print("score,r2_1,r2_2,epochs,kernel_size,points,model_param1,model_param2,telescope,t_int,label,ps_bins_to_make,ps_bins_to_use")
    
    with open(filename, 'r') as f:
        for line in f:
            result = parse_log_line(line)
            if result:
                eprint(f"{result['points']},{result['model_param1']},{result['model_param1']}")
                print(f"{result['score']},{result['r2_1']},{result['r2_2']},{result['epochs']},"
                      f"{result['kernel_size']},{result['points']},{result['model_param1']},"
                      f"{result['model_param2']},{result['telescope']},{result['t_int']},\"{result['label']}\","
                      f"{result['ps_bins_to_make']},{result['ps_bins_to_use']},")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_results.py <logfile>")
        sys.exit(1)
        
    process_file(sys.argv[1])
