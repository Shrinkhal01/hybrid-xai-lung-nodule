import pylidc as pl

# 1. Query the database for the first scan (LIDC-IDRI-0006)
scan = pl.query(pl.Scan).first()
print(f"Loaded Patient ID: {scan.patient_id}")

# 2. Cluster annotations (groups the 4 doctors' drawings together)
nodules = scan.cluster_annotations()
print(f"Found {len(nodules)} distinct nodules in this scan.")

# 3. Grab the very first annotation of the first nodule
first_nodule = nodules[0]
annotation = first_nodule[0]

# Let's see what the radiologist thought about this nodule!
print(f"Malignancy rating (1-5): {annotation.malignancy}")
print(f"Spiculation (spikiness, 1-5): {annotation.spiculation}")
print(f"Nodule Diameter: {annotation.diameter:.2f} mm")

# 4. Visualize it! 
annotation.visualize_in_scan()
