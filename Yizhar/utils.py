import os
import librosa
from typing import List


def get_files(path: str) -> List[str]:
	file_names = []
	for root,d_names,f_names in os.walk(path):
		for f in f_names:
			file_names.append(os.path.join(root, f))
	return file_names

def get_longest_duration_file(files_list: List[str]):
	return max(
		[(file, librosa.get_duration(path=file)) for file in files_list]
		, key = lambda x: x[1])
    
		