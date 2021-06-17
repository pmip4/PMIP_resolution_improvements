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
   - PMIP2 is taken from CCSM

Panel 3.
  - Now we're onto CCSM4, and can easily download the files from the ESGF
    Unfortunately, we also have to deal with rotated ocean grids, which opens a whole new stack of problems

Panel 4.
 - Finally we have CESM2 - the most up-to-date version of NCARs models.

This script was originally founded on the CB_Temperature.py script from GeogCAT_examples.
It went through an iteration that involved overlay plots of the land and ocean. This never quite worked. 

"""

###############################################################################
# Import packages:

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xesmf as xe

from geocat.viz import util as gvutil

###############################################################################
# Create a plotting script:

def Plot(topo, title, ax, inc_cbar):

    # Use the 'terrain' colormap. Add transparent to be used for NaN
    newcmp = plt.cm.get_cmap("terrain")
    #newcmp.set_bad(color=(0.1, 0.2, 0.5, 0.0))
  
    # Contourf-plot data
    tmp = topo.plot.contourf(ax=ax,
                           transform=projection,
                           levels=128,
                           vmin=-200,
                           vmax=800,
                           cmap=newcmp,
                           add_colorbar=False)

    if inc_cbar == "True":
      # Add color bar
      cbar_ticks = np.arange(-2000, 2000, 200)
      cbar = plt.colorbar(tmp,
                        orientation='vertical',
                        shrink=0.8,
                        pad=0.05,
                        extendrect=True,
                        ticks=cbar_ticks)
      cbar.ax.tick_params(labelsize=10)

    # Use geocat.viz.util convenience function to set axes parameters without calling several matplotlib functions
    # Set axes limits, and tick values
    gvutil.set_axes_limits_and_ticks(ax, xlim=(90, 140), ylim=(-20, 20))

    # Use geocat.viz.util convenience function to set titles and labels without calling several matplotlib functions
    gvutil.set_titles_and_labels(ax,
                                 maintitle=title,
                                 maintitlefontsize=14,
                                 xlabel="",
                                 ylabel="")

###############################################################################
# After xesmf documentation: https://xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html
def regrid(ds_in, ds_out, dr_in):
    """Convenience function for one-time regridding"""
    regridder = xe.Regridder(ds_in, ds_out, "nearest_s2d", periodic=True)
    dr_out = regridder(dr_in)
    return dr_out

###############################################################################
# Generate axes, using Cartopy, drawing coastlines, and adding features
projection= ccrs.PlateCarree()
fig = plt.figure(figsize=(8, 8))
grid = fig.add_gridspec(nrows=1, ncols=4, wspace=0.08)

###############################################################################
# Create an exceeding high resolution grid for plotting
ds_fine = xr.Dataset({'lat': (['lat'], np.arange(-89.75, 89.75, 0.25)),
                     'lon': (['lon'], np.arange(0,360,0.25)),
                    }
                   )

###############################################################################
# GENESIS
ax1 = fig.add_subplot(grid[0,0], projection=ccrs.PlateCarree())
gen_orog_f = xr.open_dataset("data/GENESIS_TOPO.nc",decode_times=False)
gen_orog=gen_orog_f.TOPO.isel(time=0)
gen=regrid(gen_orog_f,ds_fine,gen_orog)
Plot(gen, "PMIP1", ax1, "False")

###############################################################################
# CCSM
ax2 = fig.add_subplot(grid[0,1], projection=ccrs.PlateCarree())
ccsm_bath_f = xr.open_dataset("data/zobt_O1.PIcntrl.CCSM.ocnm.nc",decode_times=False)
ccsm_bath=regrid(ccsm_bath_f,ds_fine,ccsm_bath_f.zobt)
# For visual pleasantness, set land points in the ocean that aren't land according the atmos model as a depth of 20m  
ccsm_bath.values=np.nan_to_num(ccsm_bath.values, nan=20)
ccsm_orog_f = xr.open_dataset("data/orog_A1.PIcntrl.CCSM.atmm.nc",decode_times=False)
ccsm_orog=regrid(ccsm_orog_f,ds_fine,ccsm_orog_f.orog)
ccsm_landfrac_f = xr.open_dataset("data/sftlf_A_FX_pmip2_6k_oa_CCSM.nc",decode_times=False)
ccsm_landfrac=regrid(ccsm_landfrac_f,ds_fine,ccsm_landfrac_f.sftlf.isel(time=0))
ccsm=xr.where(ccsm_landfrac > 0.5,ccsm_orog,-ccsm_bath)
Plot(ccsm, "PMIP2", ax2, "False")

###############################################################################
# CCSM4
ax3 = fig.add_subplot(grid[0,2], projection=ccrs.PlateCarree())
ccsm4_bath_f = xr.open_dataset("data/deptho_fx_CCSM4_piControl_r0i0p0.nc",decode_times=False)
ccsm4_bath=regrid(ccsm4_bath_f,ds_fine,ccsm4_bath_f.deptho)
ccsm4_bath.values=np.nan_to_num(ccsm4_bath.values, nan=20)
ccsm4_orog_f = xr.open_dataset("data/orog_fx_CCSM4_piControl_r0i0p0.nc",decode_times=False)
ccsm4_orog=regrid(ccsm4_orog_f,ds_fine,ccsm4_orog_f.orog)
ccsm4_landfrac_f = xr.open_dataset("data/sftlf_fx_CCSM4_piControl_r0i0p0.nc",decode_times=False)
ccsm4_landfrac=regrid(ccsm4_landfrac_f,ds_fine,ccsm4_landfrac_f.sftlf)
ccsm4=xr.where(ccsm4_landfrac > 50.,ccsm4_orog,-ccsm4_bath)
Plot(ccsm4, "PMIP3", ax3, "False")

###############################################################################
# CESM2
ax4 = fig.add_subplot(grid[0,3], projection=ccrs.PlateCarree())
cesm2_bath_f = xr.open_dataset("data/deptho_Ofx_CESM2_piControl_r1i1p1f1_gn.nc",decode_times=False)
cesm2_bath=regrid(cesm2_bath_f,ds_fine,cesm2_bath_f.deptho)
cesm2_bath.values=np.nan_to_num(cesm2_bath.values, nan=20)
cesm2_orog_f = xr.open_dataset("data/orog_fx_CESM2_piControl_r1i1p1f1_gn.nc",decode_times=False)
cesm2_orog=regrid(cesm2_orog_f,ds_fine,cesm2_orog_f.orog)
cesm2_landfrac_f = xr.open_dataset("data/sftlf_fx_CESM2_midHolocene_r1i1p1f1_gn.nc",decode_times=False)
cesm2_landfrac=regrid(cesm2_landfrac_f,ds_fine,cesm2_landfrac_f.sftlf)
cesm2=xr.where(cesm2_landfrac > 50.,cesm2_orog,-cesm2_bath)
Plot(cesm2, "PMIP4", ax4, "False")

fig.savefig('output.pdf')