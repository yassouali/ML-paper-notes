#!/usr/bin/python3

import os, sys
import argparse
import re
from googlesearch import search
from shutil import copytree, copy
from subprocess import call
from pathlib import Path
import pickle
from tqdm import tqdm

DOCUMENTS_PATH = "/home/yassine/Writing/papers_notes"
r_match = r'\\title{ \\LARGE \\textbf{(.*?)}'
match_year = r'} \\\\ (.*?)}'

# Get the committed papers to avoid duplication
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
		search_results = []
		for p, _ in tqdm(new_papers):
			res = search(p, tld="co.in", num=10, stop=5, pause=2)
			for link in res:
				if "arxiv" in link:
					search_results.append(link)
					break
		search_success = True
	except:
		pass
paper_name_link = {n:link for link, n in zip(search_results, new_papers)}

# Append the new papers to the markdown file
# with open('README.md', 'a') as f:
# 	for (n, path), link in paper_name_link.items():
# 		new_line = f'- {n.rstrip()}: [[Paper]]({link}) [[Notes]]({path})'
# 		f.write(f'{new_line}\n')

f = open("README.md", "r")
contents = f.readlines()
f.close()

position = 5
for (n, path), link in paper_name_link.items():
	new_line = f'- {n.rstrip()}: [[Paper]]({link}) [[Notes]]({path})'
	contents.insert(position, f'{new_line}\n')

f = open("README.md", "w")
contents = "".join(contents)
f.write(contents)
f.close()

# Saving a new list of the committed papers
commited_papers += [p for p, _ in new_papers]
pickle.dump(commited_papers, open('commited_papers.pkl', 'wb'))

# Git add and commit
os.system("git add .")
os.system(f"git commit -m 'New commit, added {len(new_papers)} papers'")
os.system("git push")
