import os
import numpy as np
import MDAnalysis as mda

# --------------------------------------------------------------------------------
# This script illustrates the ordering of the electric field sampling positions
# around the NH bond to use in the NH stretch map. The electric fields MUST be
# inputted in this exact order for the map to work
# --------------------------------------------------------------------------------
uni = mda.Universe("ala3.pdb")  # sample pdb file containing a protein
nPosition = uni.select_atoms("resid 3 and name N").positions[0]  # sample NH group, "N" atom
hPosition = uni.select_atoms("resid 3 and name H").positions[0]  # sample NH group, "H" atom
cPosition = uni.select_atoms("resid 2 and name C").positions[0]  # corresponding carbonyl "C" atom

nHBond = hPosition - nPosition  # define NH bond ("x")
cNBond = cPosition - nPosition  # define NC bond

nHBond /= np.linalg.norm(nHBond)  # normalize NH bond
cNBond /= np.linalg.norm(cNBond)  # normalize NC bond

perp = cNBond - np.dot(cNBond, nHBond) * nHBond  # generate vector pointing somewhat along NC bond but perpedicular to NH ("y")
perp /= np.linalg.norm(perp)  # normalize this perpendicular vector

perp2 = np.cross(nHBond, perp)  # generate a third vector ("z") perpendicular to both intial vectors
perp2 /= np.linalg.norm(perp2)  # normalize the third vector
# keep in mind, the cross product should be in the opposite order (left-handed) for D-enantiomers

# generate three positions along the NH bond ("x") axis, starting with the nitrogen atom
pos1 = nPosition + nHBond * 1.0
pos2 = nPosition + nHBond * 1.5
pos3 = nPosition + nHBond * 2.0
poss = [pos1, pos2, pos3]

# define a displacement of 0.2 Å from the NH bond axis at each position
disp = 0.2
# define "up" "down" "right" "left" vectors
up = disp * perp
down = -disp * perp
right = disp * perp2
left = -disp * perp2

# define positions along the NH bond
positions = [pos1, pos2, pos3]

# define positions 0.2 Å from the NH bond axis in all directions, starting from each of three positions
positions += [pos1 + up, pos1 + down, pos1 + right, pos1 + left,
              pos1 + right + up, pos1 + right + down, pos1 + left + up, pos1 + left + down]
positions += [pos2 + up, pos2 + down, pos2 + right, pos2 + left,
              pos2 + right + up, pos2 + right + down, pos2 + left + up, pos2 + left + down]
positions += [pos3 + up, pos3 + down, pos3 + right, pos3 + left,
              pos3 + right + up, pos3 + right + down, pos3 + left + up, pos3 + left + down]

# define a displacement of 0.1 Å from the NH bond axis at each position
disp = 0.1
up = disp * perp
down = -disp * perp
right = disp * perp2
left = -disp * perp2

# define positions 0.1 Å from the NH bond axis in all directions, starting from each of three positions
positions += [pos1 + up, pos1 + down, pos1 + right, pos1 + left,
              pos1 + right + up, pos1 + right + down, pos1 + left + up, pos1 + left + down]
positions += [pos2 + up, pos2 + down, pos2 + right, pos2 + left,
              pos2 + right + up, pos2 + right + down, pos2 + left + up, pos2 + left + down]
positions += [pos3 + up, pos3 + down, pos3 + right, pos3 + left,
              pos3 + right + up, pos3 + right + down, pos3 + left + up, pos3 + left + down]

# --------------------------------------------------------------------------------
# the positions vector now has a defined ordering -> use this ordering for the map
# just one more thing: for each position, sample three electric fields along "x", "y",
# and "z" as defined above, so the final ordering of electric fields is:
# pos1_x pos1_y pos1_z pos2_x pos2_y pos2_z ...
# --------------------------------------------------------------------------------
arrowPos = positions[-1]
with open("dots.bild", "w+") as f:
    for pos in positions:
        f.write(".color red\n")
        f.write(f".sphere {pos[0]} {pos[1]} {pos[2]} 0.02\n")
    x = arrowPos[0]
    y = arrowPos[1]
    z = arrowPos[2]
    f.write(".color black\n")
    f.write(f".arrow {x} {y} {z} {x + nHBond[0] / 2} {y + nHBond[1] / 2} {z + nHBond[2] / 2} 0.01 \n")
    f.write(".color black\n")
    f.write(f".arrow {x} {y} {z} {x + perp[0] / 2} {y + perp[1] / 2} {z + perp[2] / 2} 0.01\n")
    f.write(".color black\n")
    f.write(f".arrow {x} {y} {z} {x + perp2[0] / 2} {y + perp2[1] / 2} {z + perp2[2] / 2} 0.01\n")

os.system("/Applications/Chimera.app/Contents/MacOS/chimera ala3.pdb dots.bild")
