#!/bin/bash
wget -O ./exp-referit/referit-dataset/referitdata.tar.gz http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referitdata.tar.gz
tar -xzvf ./exp-referit/referit-dataset/referitdata.tar.gz -C ./exp-referit/referit-dataset/
wget -O ./exp-referit/data/referit_edgeboxes_top100.zip http://www.eecs.berkeley.edu/~ronghang/projects/cvpr16_text_obj_retrieval/referit_edgeboxes_top100.zip
unzip ./exp-referit/data/referit_edgeboxes_top100.zip -d ./exp-referit/data/
