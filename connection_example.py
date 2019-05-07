from phase.connections import count_connections

# We will first run `count_connections` on the homogeneous systems
filepath = '/raid6/homes/raymat/science/keras-phase-sep/data-otsu/train/homo'
labeled_path = 'data-labeled'
count_connections(filepath, labeled_path)


print('Looking at heterogeneous systems now')
# Now we will run `count_connections` on the heterogeneous systems
filepath = '/raid6/homes/raymat/science/keras-phase-sep/data-otsu/train/hetero'
count_connections(filepath, labeled_path)
