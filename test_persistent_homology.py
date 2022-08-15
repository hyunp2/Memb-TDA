import persim
import ripser
import MDAnalysis as mda

u_open = mda.fetch_mmtf('3CLN')
u_inter = mda.fetch_mmtf('1CFD')
u_closed = mda.fetch_mmtf('1L7Z')
Pa1 = u_open.select_atoms("backbone and segid A")
Pa2 = u_inter.select_atoms("backbone and segid A")
Pa3 = u_closed.select_atoms("backbone and segid A")

R1 = ripser.ripser(Pa1.atoms.positions)['dgms'][1]
R2 = ripser.ripser(Pa2.atoms.positions)['dgms'][1]
R3 = ripser.ripser(Pa3.atoms.positions)['dgms'][1]

print(persim.wasserstein(R1, R2))
print(persim.wasserstein(R2, R3))
print(persim.wasserstein(R1, R3))
