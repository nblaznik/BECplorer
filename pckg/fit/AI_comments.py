import os
import re

def combine_all_files(directory, output_file):
    files = [f for f in os.listdir(directory) if re.match(r'\d{8}_\d+\.txt$', f)]
    files.sort()  # Sort by date naturally
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file in files:
            date_part = file.split('_')[0]
            number_part = file.split('_')[1].split('.')[0]
            file_path = os.path.join(directory, file)
            
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read().strip()
                outfile.write(f"{date_part} - {number_part}:   {content}\n")

if __name__ == "__main__":
    directory = "/storage/BECViewer/comments"  # Change this to your actual folder path
    output_file = "all_comments.txt"
    
    combine_all_files(directory, output_file)
    print(f"Combined file saved as {output_file}")