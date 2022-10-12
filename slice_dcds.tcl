
#NOTE MAKE INTO PROC
set NUM [lindex $argv 0]

mol new /Scr/arango/Sobolev-Hyun/2-MembTempredict/DPPC_$NUM/namd/test2.psf
mol addfile /Scr/arango/Sobolev-Hyun/2-MembTempredict/DPPC_$NUM/namd/test2.dcd waitfor all

set num_frames [molinfo 0 get numframes]

#get_cell based on water box, avoiding issues with lipids
for {set fram 0} {$fram < $num_frames} {incr fram} {
	animate goto $fram
	set molid "top"
	 set all [atomselect $molid "all" frame now]
	 set minmax [measure minmax $all]
	 set vec [vecsub [lindex $minmax 1] [lindex $minmax 0]]
	
	 set cellX [lindex $vec 0] 
	 set cellY [lindex $vec 1] 
	 #set center [measure center $all]
	 #set cellBasisVectorO "cellOrigin $center"
	 $all delete
	set slice 3
	set keeper 0
	for {set i -$slice} {$i < [expr $slice -1]} {incr i} {
		for {set j -$slice} {$j < [expr $slice -1]} {incr j} {
			incr keeper
			set selcube [atomselect top "same residue as x > [expr $cellX/2 * $i/$slice ] and y > [expr $cellY /2* $j / $slice] and x < [expr $cellX/2 * ($i+1)/$slice ] and y < [expr $cellY /2* ($j+1)/ $slice]" frame now] 
			#mol selection same residue as x > [expr $cellX/2 * $i/$slice ] and y > [expr $cellY /2* $j / $slice] and x < [expr $cellX/2 * ($i+1)/$slice ] and y < [expr $cellY /2* ($j+1)/ $slice] 
		
			#mol addrep 0 
			#mol modstyle $keeper 0 VDW 1.000000 12.000000
			#puts "Atom count per slice:"
			#puts [$selcube num]
			$selcube writepdb $fram.$keeper.pdb
		}
	}
}
exit
