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
from numpy.lib.function_base import copy
import xarray as xr

from geocat.viz import cmaps as gvcmaps
from geocat.viz import util as gvutil

###############################################################################
# Create a plotting script:

def Plot(topo, isOro, title, ax, inc_cbar):

    # Use the 'terrain' colormap. Add transparent to be used for NaN
    newcmp = plt.cm.get_cmap("terrain")
    newcmp.set_bad(color=(0.1, 0.2, 0.5, 0.0))
  
    if isOro == "True":
      topo = topo.where(topo > 0.0)
      order = 2
    else:
      topo = 1-topo #flip bathymetry to negative
      order = 1  


    # Contourf-plot data
    tmp = topo.plot.pcolormesh(ax=ax,
                           transform=projection,
                           levels=128,
                           vmin=-500,
                           vmax=2000,
                           cmap=newcmp,
                           add_colorbar=False,
                           zorder=order)

    if inc_cbar == "True":
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
    gvutil.set_axes_limits_and_ticks(ax, xlim=(90, 120), ylim=(-20, 20))

    # Use geocat.viz.util convenience function to set titles and labels without calling several matplotlib functions
    gvutil.set_titles_and_labels(ax,
                                 maintitle=title,
                                 maintitlefontsize=14,
                                 xlabel="",
                                 ylabel="")


###############################################################################
# Create a plotting script:

def Plot_rotated(topo, lat2d, lon2d, isOro, title, ax):

    ax = plt.axes(projection=ccrs.PlateCarree())

    # Use the 'terrain' colormap. Add transparent to be used for NaN
    newcmp = plt.cm.get_cmap("terrain")
    newcmp.set_bad(color=(0.1, 0.2, 0.5, 0.0))
  
    if isOro == "True":
      topo = topo.where(topo > 0.0)
      order = 2
    else:
      topo = 1-topo #flip bathymetry to negative
      order = 1  


    # Contourf-plot data
    tmp = plt.pcolormesh( lat2d, lon2d, topo,
                           #transform=projection,
                           vmin=-500,
                           vmax=2000,
                           cmap=newcmp,
                           zorder=order)

    # Use geocat.viz.util convenience function to set axes parameters without calling several matplotlib functions
    # Set axes limits, and tick values
    gvutil.set_axes_limits_and_ticks(ax, xlim=(90, 120), ylim=(-20, 20))

    # Use geocat.viz.util convenience function to set titles and labels without calling several matplotlib functions
    gvutil.set_titles_and_labels(ax,
                                 maintitle=title,
                                 maintitlefontsize=14,
                                 xlabel="",
                                 ylabel="")


###############################################################################
# Generate axes, using Cartopy, drawing coastlines, and adding features
projection= ccrs.PlateCarree()
fig = plt.figure(figsize=(8, 8))
grid = fig.add_gridspec(nrows=1, ncols=4, wspace=0.08)

###############################################################################
# GENESIS
ax1 = fig.add_subplot(grid[0,0], projection=ccrs.PlateCarree())
gen_bath_f = xr.open_dataset("data/GENESIS_LANDFRAC.nc",decode_times=False)
gen_bath=gen_bath_f.LANDFRAC.where(gen_bath_f.LANDFRAC < 0.5)
gen_bath=gen_bath+100
gen_orog_f = xr.open_dataset("data/GENESIS_TOPO.nc",decode_times=False)
gen_orog=gen_orog_f.TOPO.where(gen_orog_f.TOPO > 0.001)
Plot(gen_bath.isel(time=0), "False", "PMIP1", ax1, "False")
Plot(gen_orog.isel(time=0), "True", "PMIP1", ax1, "False")

###############################################################################
# CCSM
ax2 = fig.add_subplot(grid[0,1], projection=ccrs.PlateCarree())
# Open a netCDF data file using xarray default engine and load the data into xarrays
ccsm_bath_f = xr.open_dataset("data/zobt_O1.PIcntrl.CCSM.ocnm.nc",decode_times=False)
ccsm_bath=ccsm_bath_f.zobt
ccsm_orog_f = xr.open_dataset("data/orog_A1.PIcntrl.CCSM.atmm.nc",decode_times=False)
ccsm_orog=ccsm_orog_f.orog.where(ccsm_orog_f.orog > 0.001)
Plot(ccsm_bath, "False", "PMIP2", ax2, "False")
Plot(ccsm_orog, "True", "PMIP2", ax2, "False")

###############################################################################
# CCSM4
ax3 = fig.add_subplot(grid[0,2], projection=ccrs.PlateCarree())
ccsm4_bath_f = xr.open_dataset("data/deptho_fx_CCSM4_piControl_r0i0p0.nc",decode_times=False)
ccsm4_bath=ccsm4_bath_f.deptho
ccsm4_orog_f = xr.open_dataset("data/orog_fx_CCSM4_piControl_r0i0p0.nc",decode_times=False)
ccsm4_orog=ccsm4_orog_f.orog.where(ccsm4_orog_f.orog > 0.001)
Plot_rotated(ccsm4_bath, ccsm4_bath_f.lat, ccsm4_bath_f.lon, "False", "PMIP3", ax3)
Plot(ccsm4_orog, "True", "PMIP3", ax3, "False")

###############################################################################
# CESM2
ax4 = fig.add_subplot(grid[0,3], projection=ccrs.PlateCarree())
cesm2_bath_f = xr.open_dataset("data/deptho_Ofx_CESM2_piControl_r1i1p1f1_gn.nc",decode_times=False)
cesm2_bath=cesm2_bath_f.deptho
cesm2_orog_f = xr.open_dataset("data/orog_fx_CESM2_piControl_r1i1p1f1_gn.nc",decode_times=False)
cesm2_orog=cesm2_orog_f.orog.where(cesm2_orog_f.orog > 0.001)
Plot_rotated(cesm2_bath, cesm2_bath_f.lat, cesm2_bath_f.lon, "False", "PMIP4", ax4)
Plot(cesm2_orog, "True", "PMIP4", ax4, "True")

fig.savefig('outplot.pdf')