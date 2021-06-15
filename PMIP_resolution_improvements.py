"""
PMIP_resolution_improvements.py
=================

This script illustrates the changes in resolution that have occurred across the various generations of PMIP.
The original idea is based on an infrographic by Nicola Jones in Nature (2014, doi:10.1038/501298a).
It uses model grids from various NCAR models, explained below. The topography is plotted over the land,
whilst the bathymetry is plotted over the ocean.

Panel 1.
   - This panel plots the NCAR Genesis model, which had a T31 resolution. 
     The topography is contained in the GENESIS_PHIS.nc file, which needs to be converted to metres by dividing by gravity.
     PMIP1 was an atmosphere-only experiment, so there is no bathymetry plotted. 
     [Note: I couldn't find the required file on the PMIP1 database. So this file actually 
     comes from the CCSM4 T31 palaeoclimate resolution, which is very similar].

Panel 2.
   - The coolwarm diverging scheme should be used when both high and low values are interesting.
     However, be careful using this scheme if the projection will be printed to black and white.

Panel 3.
  - This is an example of a less distinct contrasting color gradient. This choice in color scheme would
    be a good choice for printing in black and white but may create some challenges for individuals who
    experience blue-green colorblindness.

Panel 4.
 - This plot shows how drastically contrasting colors can be incredibly useful for plotting this type of data.
   This color scheme will work well for color blind impacted individuals and is black and white print friendly.

This script was originally based on the CB_Temperature.py script from GeogCAT_examples.

"""

###############################################################################
# Import packages:

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil

###############################################################################
# Create a plotting script:

def Plot(topo, isOro, pos, title):

    # Generate axes, using Cartopy, drawing coastlines, and adding features
    projection = ccrs.PlateCarree()
    ax1 = plt.subplot(1, pos, pos, projection=projection)
    # Use the 'terrain' colormap

    newcmp = copy.copy(plt.cm.get_cmap("terrain"))
    newcmp.set_bad(color=(0.1, 0.2, 0.5, 0.0))
  
    if isOro == "True":
      topo = topo.where(topo > 0.0)
      order = 2
    else:
      topo = 1-topo #flip bathymetry to negative
      order = 1  


    # Contourf-plot data
    tmp = topo.plot.pcolormesh(ax=ax1,
                           transform=projection,
                           levels=128,
                           vmin=-500,
                           vmax=2000,
                           cmap=newcmp,
                           add_colorbar=False,
                           zorder=order)

    if isOro == "True":
      # Add color bar
      cbar_ticks = np.arange(-2000, 2000, 250)
      cbar = plt.colorbar(tmp,
                        orientation='vertical',
                        shrink=0.8,
                        pad=0.05,
                        extendrect=True,
                        ticks=cbar_ticks)
      cbar.ax.tick_params(labelsize=10)

    # Use geocat.viz.util convenience function to set axes parameters without calling several matplotlib functions
    # Set axes limits, and tick values
    gvutil.set_axes_limits_and_ticks(ax1, xlim=(90, 120), ylim=(-20, 20))

    # Use geocat.viz.util convenience function to set titles and labels without calling several matplotlib functions
    gvutil.set_titles_and_labels(ax1,
                                 maintitle=title,
                                 maintitlefontsize=14,
                                 xlabel="",
                                 ylabel="")


###############################################################################
# Open and set up the plot:

fig = plt.figure(figsize=(16, 4))

###############################################################################
# GENESIS
gen_bath_f = xr.open_dataset("data/GENESIS_LANDFRAC.nc",decode_times=False)
gen_bath=gen_bath_f.LANDFRAC.where(gen_bath_f.LANDFRAC < 0.5)
gen_bath=gen_bath+100
gen_orog_f = xr.open_dataset("data/GENESIS_TOPO.nc",decode_times=False)
gen_orog=gen_orog_f.TOPO.where(gen_orog_f.TOPO > 0.001)
#Plot(gen_bath.isel(time=0), "False", 1, "PMIP1")
#Plot(gen_orog.isel(time=0), "True", 1, "PMIP1")

###############################################################################
# CCSM

# Open a netCDF data file using xarray default engine and load the data into xarrays
ccsm_bath_f = xr.open_dataset("data/zobt_O1.PIcntrl.CCSM.ocnm.nc",decode_times=False)
ccsm_bath=ccsm_bath_f.zobt
ccsm_orog_f = xr.open_dataset("data/orog_A1.PIcntrl.CCSM.atmm.nc",decode_times=False)
ccsm_orog=ccsm_orog_f.orog.where(ccsm_orog_f.orog > 0.001)
#Plot(ccsm_bath, "False", 2, "PMIP1")
#Plot(ccsm_orog, "True", 2, "PMIP2")

###############################################################################
# CCSM4

# Open a netCDF data file using xarray default engine and load the data into xarrays
ccsm4_bath_f = xr.open_dataset("data/deptho_fx_CCSM4_piControl_r0i0p0.nc",decode_times=False)
ccsm4_bath=ccsm4_bath_f.deptho
ccsm4_orog_f = xr.open_dataset("data/orog_fx_CCSM4_piControl_r0i0p0.nc",decode_times=False)
ccsm4_orog=ccsm4_orog_f.orog.where(ccsm4_orog_f.orog > 0.001)
Plot(ccsm4_bath, "False", 3, "PMIP3")
#Plot(ccsm4_orog, "True", 3, "PMIP3")

###############################################################################
# CESM2
cesm2_bath_f = xr.open_dataset("data/deptho_Ofx_CESM2_piControl_r1i1p1f1_gn.nc",decode_times=False)
cesm2_bath=cesm2_bath_f.deptho
cesm2_orog_f = xr.open_dataset("data/orog_fx_CESM2_piControl_r1i1p1f1_gn.nc",decode_times=False)
cesm2_orog=cesm2_orog_f.orog.where(cesm2_orog_f.orog > 0.001)
#Plot(cesm2_bath, "False", 4, "PMIP4")
#Plot(cesm2_orog, "True", 4, "PMIP4")

#fig.suptitle("Projections of Temperature", x=.5, y=.95, fontsize=18)
fig.savefig('foo.png')