import os

# Define the main directory path containing subfolders
main_directory = '/home/sallyjunsongwang/nvbit_release/tools/nvbitfi/logs/2way_memory_conflict'

# The target 'global sum' value to check against
TARGET_GLOBAL_SUM = 2096128

# Function to process each subfolder
def process_subfolder(subfolder_path):
    stdout_file_path = os.path.join(subfolder_path, 'stdout.txt')
    injection_log_file_path = os.path.join(subfolder_path, 'nvbitfi-injection-log-temp.txt')
    stderr_file_path = os.path.join(subfolder_path, 'stderr.txt')

    # Check if stdout.txt exists in the subfolder
    if os.path.isfile(stdout_file_path):
        with open(stdout_file_path, 'r') as stdout_file:
            # Read lines and find the line with 'global sum'
            for line in stdout_file:
                if 'global sum' in line:
                    # Extract the global sum value
                    global_sum = int(line.split('=')[1].strip())
                    
                    # Calculate the difference from the target
                    difference = global_sum - TARGET_GLOBAL_SUM

                    # If global sum is not equal to the target, read the injection log file
                    if global_sum != TARGET_GLOBAL_SUM:
                        # Print the difference in global sum
                        print(f"Subfolder: {subfolder_path}, Global Sum Difference: {difference}")

                        # Check if nvbitfi-injection-log-temp.txt exists
                        if os.path.isfile(injection_log_file_path):
                            with open(injection_log_file_path, 'r') as injection_log_file:
                                # Initialize variables to store values
                                mask, beforeVal, afterVal, opcode = None, None, None, None

                                # Iterate through the lines to find the desired values
                                for log_line in injection_log_file:
                                    if 'opcode' in log_line:
                                        opcode = log_line.split(':')[1].strip()
                                    if 'mask' in log_line:
                                        mask = log_line.split(':')[1].strip()
                                    if 'beforeVal' in log_line:
                                        beforeVal = log_line.split(':')[1].strip()
                                    if 'afterVal' in log_line:
                                        afterVal = log_line.split(';')[1].strip()

                                # Print the extracted values
                                print(f"Opcode: {opcode}, Mask: {mask}, BeforeVal: {beforeVal.split(';')[0].strip()}, AfterVal: {afterVal.split(':')[1].strip()}")

                                # Check if stderr.txt exists and is not empty
                                if os.path.isfile(stderr_file_path) and os.path.getsize(stderr_file_path) > 0:
                                    print(f"DUE - Opcode: {opcode}")
                        else:
                            print(f"'nvbitfi-injection-log-temp.txt' not found in {subfolder_path}.")
                    break
    else:
        print(f"'stdout.txt' not found in {subfolder_path}.")

# Main function to iterate through subfolders in the main directory
def main():
    # Check if the main directory exists
    if not os.path.exists(main_directory):
        print(f"The directory '{main_directory}' does not exist.")
        return

    # Iterate through each subfolder in the main directory
    for subfolder in os.listdir(main_directory):
        subfolder_path = os.path.join(main_directory, subfolder)
        
        # Process only directories
        if os.path.isdir(subfolder_path):
            process_subfolder(subfolder_path)

if __name__ == "__main__":
    main()
