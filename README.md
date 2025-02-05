### Running instructions:

1. Clone the repo on COARE
2. Upload/Move the already cropped dataset and the generated annotations file (index.csv) to a location on the scratch folder
3. Edit classifier.py and specify the ANNOTATIONS_FILE and IMG_DIR filepaths
4. Create the environment with `sbatch create-env.slurm`
5. Run the classifier with `sbatch run-classifier.slurm`. The output will be in `run-classifier.out` once the model is finished training