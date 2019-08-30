#!/usr/bin/python3

import os
import argparse
import re
from google import google
from shutil import copytree, copy
from subprocess import call
from pathlib import Path
import pickle
from tqdm import tqdm

DOCUMENTS_PATH = "/home/yassine/Documents/papers_notes/"
r_match = r'\\title{ \\LARGE \\textbf{(.*?)}'
match_year = r'} \\\\ (.*?)}'

# Get the commited papers to avoid duplication
try:
	commited_papers = pickle.load(open("commited_papers.pkl", "rb"))
except:
	commited_papers = []

# Get the names of the papers
dir_path = Path(DOCUMENTS_PATH)
folders = [i for i in dir_path.iterdir() if i.name.split('_')[0].isdigit()]
tex_files = {list(f.glob('*.tex'))[0]:list(f.glob('*.pdf'))[0] for f in folders}
papers_title = {re.findall(r_match, open(t).read(), re.S)[0].replace('\n', ' '):pdf for t, pdf in tex_files.items()}
papers_year = [re.findall(match_year, open(t).read(), re.S)[0] for t, _ in tex_files.items()]
papers_title = {(i+' '+k):j for (i,j), k in zip(papers_title.items(), papers_year)}
new_papers = [p for p in papers_title.items() if p[0] not in commited_papers]

# Copy new notes into this git dir
cwd = Path.cwd()
for i, (n, path) in enumerate(new_papers):
	new_path = cwd / Path(f'notes/{path.name}')
	copy(path, new_path)
	new_papers[i] = (n, f'notes/{path.name}')

# Get the paper link
search_success = False
while not search_success:
	try:
		search_results = [google.search(p, 1)[0] for p, _ in tqdm(new_papers)]
		search_success = True
	except:
		pass
paper_name_link = {n:s.link for s, n in zip(search_results, new_papers)}

# Append the new papers to the markdown file
with open('README.md', 'a') as f:
	for (n, path), link in paper_name_link.items():
		new_line = f'- {n.rstrip()}: [[Paper]]({link}) [[Notes]]({path})'
		f.write(f'{new_line}\n')

# Git add and commit
os.system("git pull")
os.system("git add .")
os.system(f"git commit -m 'New commit, added {len(new_papers)} papers'")
os.system("git push")

# Saving a new list of the commited papers
commited_papers += [p for p, _ in new_papers]
pickle.dump(commited_papers, open('commited_papers.pkl', 'wb'))


