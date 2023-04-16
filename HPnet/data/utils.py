# import os
# import shutil

# import torch
# import torch.utils.data
# import transforms
# import torchvision.datasets as datasets
# import argparse
# from helpers import makedir, adjust_learning_rate
# import model
# import push
# import train_and_test as tnt
# import save
# from log import create_logger
# from preprocess import mean, std, preprocess_input_function, img_size
from node import Node
# import time
# import numpy as np
from data.phylogeny import PhylogenyCUB

# def construct_phylo_tree(phylogeny_path, phyloDistances_string):
#     phylo = PhylogenyCUB(phylogeny_path) # '/home/harishbabu/data/phlyogenyCUB'
#     root = Node("root")
#     phyloDistances = [float(x) for x in phyloDistances_string.split(',')[::-1]] + [1]
#     num_levels = len(phyloDistances)

#     ances_lvl_tag_prefix = '_lvl'

#     ancestor_lvl_to_spc_groups = {} # maps ancestor levels (int) to spc groups (dict mapping representative_species to a list of species)
#     for ancestor_lvl, phylo_dist in enumerate(phyloDistances[:-1]):
#         ancestor_lvl_to_spc_groups[ancestor_lvl] = {(spc_group[0] + ances_lvl_tag_prefix + str(ancestor_lvl)): spc_group \
#                                                     for spc_group in phylo.get_species_groups(1-phylo_dist)}
#         if ancestor_lvl == 0:
#             children_list = []
#             for representative, spc_group in ancestor_lvl_to_spc_groups[ancestor_lvl].items():
#                 children_list.append(representative)
#             root.add_children(children_list)
#         else:
#             prev_level_representatives = [representative for representative, spc_group in ancestor_lvl_to_spc_groups[ancestor_lvl - 1].items()]
#             prev_level_representative_to_children = {representative: [] for representative in prev_level_representatives}
#             for representative, spc_group in ancestor_lvl_to_spc_groups[ancestor_lvl].items():
#                 for prev_lvl_rep in prev_level_representatives:
#                     if representative.split(ances_lvl_tag_prefix)[0] in ancestor_lvl_to_spc_groups[ancestor_lvl - 1][prev_lvl_rep]:
#                         prev_level_representative_to_children[prev_lvl_rep].append(representative)
#                         break
            
#             for prev_lvl_rep, children in prev_level_representative_to_children.items():
#                 root.add_children_to(prev_lvl_rep, children)

#     return root


def construct_phylo_tree(phylogeny_path, phyloDistances_string):
    phylo = PhylogenyCUB(phylogeny_path) # '/home/harishbabu/data/phlyogenyCUB'
    root = Node("root")
    phyloDistances = [float(x) for x in phyloDistances_string.split(',')[::-1]] + [1]
    num_levels = len(phyloDistances)

    ances_lvl_tag_prefix = '_lvl'

    ancestor_lvl_to_spc_groups = {} # maps ancestor levels (int) to spc groups (dict mapping representative_species to a list of species)
    for ancestor_lvl, phylo_dist in enumerate(phyloDistances):
        if ancestor_lvl == len(phyloDistances)-1:
            ancestor_lvl_to_spc_groups[ancestor_lvl] = {spc_group[0]: spc_group \
                                                        for spc_group in phylo.get_species_groups(1-phylo_dist)}
        else:
            ancestor_lvl_to_spc_groups[ancestor_lvl] = {(spc_group[0] + ances_lvl_tag_prefix + str(ancestor_lvl)): spc_group \
                                                        for spc_group in phylo.get_species_groups(1-phylo_dist)}
        if ancestor_lvl == 0:
            children_list = []
            for representative, spc_group in ancestor_lvl_to_spc_groups[ancestor_lvl].items():
                children_list.append(representative)
            root.add_children(children_list)
        else:
            prev_level_representatives = [representative for representative, spc_group in ancestor_lvl_to_spc_groups[ancestor_lvl - 1].items()]
            prev_level_representative_to_children = {representative: [] for representative in prev_level_representatives}
            for representative, spc_group in ancestor_lvl_to_spc_groups[ancestor_lvl].items():
                for prev_lvl_rep in prev_level_representatives:
                    if representative.split(ances_lvl_tag_prefix)[0] in ancestor_lvl_to_spc_groups[ancestor_lvl - 1][prev_lvl_rep]:
                        prev_level_representative_to_children[prev_lvl_rep].append(representative)
                        break
            
            for prev_lvl_rep, children in prev_level_representative_to_children.items():
                root.add_children_to(prev_lvl_rep, children)

    return root



if __name__ == '__main__':
    root = construct_phylo_tree(phylogeny_path='/home/harishbabu/data/phlyogenyCUB/1_tree-consensus-Hacket-AllSpecies-modified_cub-names_v1.phy',
                          phyloDistances_string='0.93,0.83,0.63')
    print(getattr(root, 'children_names')())
