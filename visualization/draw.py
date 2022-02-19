import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gc
import sys
import matplotlib.patches as mpatches



def draw_cat(path,  label1):

    df1 = pd.read_csv(path, sep=';')
    opath = "vis/"
    #print(df1)
    #df1 = df1.loc[df1['cluster_count'] != '32']
    df1 = df1.loc[df1['benchmark'] != 'bench1_base_variant']
    df1 = df1.loc[df1['benchmark'] != 'bench2_base_variant_skipping']
    df1 = df1.loc[df1['benchmark'] != 'bench3_3pass_streaming']
    df1 = df1.loc[df1['benchmark'] != 'bench4_3pass_optimized_read_non_skipping_cub_pss']
    df1 = df1.loc[df1['benchmark'] != 'bench5_3pass_optimized_read_skipping_partial_pss']
    df1 = df1.loc[df1['benchmark'] != 'bench6_3pass_optimized_read_skipping_two_phase_pss']
    df1 = df1.loc[df1['benchmark'] != 'bench7_3pass_optimized_read_skipping_cub_pss']
    #df1 = df1.groupby(['selectivity','benchmark']).agg({'time_total': 'min'})

    df2= df1.groupby(['selectivity','benchmark']).agg({'throughput': 'max'})
    df2=df2.rename(columns={'selectivity': 'selectivity2'},
                   index={'bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss': 'SPACE_10'})
    df2 = df2.rename(columns={'selectivity': 'selectivity2'},
                     index={'bench8_cub_flagged': 'Cub::Flagged'})

    sns_plot= sns.relplot(x="selectivity",ci=None, y="throughput",hue='benchmark', kind="line", data=df2,
                         dashes=False, markers=True,style="benchmark").set(title=label1)

    cub_value_start = df2['throughput'].iloc[1]
    space_value_start = df2['throughput'].iloc[0]
    cub_value_end = df2.tail(2)
    cub_value_end =cub_value_end['throughput'].iloc[1]
    space_value_end = df2.tail(2)
    space_value_end = space_value_end['throughput'].iloc[0]
    speedup_start = space_value_start/cub_value_start
    speedup_end = space_value_end/cub_value_end
    print(speedup_start)
    print(speedup_end)


    #for ax in sns_plot.axes.flat:

      # ax.set_xscale('log')
       # ax.vlines( ymin=cub_value_start, ymax=space_value_start, x=0.0, linewidth=2, color='black',linestyles ="dashed" )
      #  ax.vlines(ymin=cub_value_end, ymax=space_value_end, x=0.97, linewidth=2, color='black',linestyles ="dashed",
      #            label="Speedup start: "+str("{:1.2f}".format(speedup_start))+"x  end: " +str("{:1.2f}".format(speedup_end))+"x")

    sns_plot._legend.remove()
    plt.legend(loc=1)
    sns_plot.savefig(opath + label + ".png")
    plt.savefig(opath + label + ".pdf", bbox_inches='tight')


    return sns_plot

def draw_grid(path,  axes, pos,label1=True, multidim=False, pos2=0, log=False, intro=False, logy=False):

    df1 = pd.read_csv(path, sep=';')
    opath = "vis/"
    #print(df1)
    #df1 = df1.loc[df1['cluster_count'] != '32']
    df1 = df1.loc[df1['benchmark'] != 'bench1_base_variant']
    df1 = df1.loc[df1['benchmark'] != 'bench2_base_variant_skipping']
    df1 = df1.loc[df1['benchmark'] != 'bench3_3pass_streaming']
    df1 = df1.loc[df1['benchmark'] != 'bench4_3pass_optimized_read_non_skipping_cub_pss']
    df1 = df1.loc[df1['benchmark'] != 'bench5_3pass_optimized_read_skipping_partial_pss']
    df1 = df1.loc[df1['benchmark'] != 'bench6_3pass_optimized_read_skipping_two_phase_pss']
    df1 = df1.loc[df1['benchmark'] != 'bench7_3pass_optimized_read_skipping_cub_pss']
    #df1 = df1.groupby(['selectivity','benchmark']).agg({'time_total': 'min'})

    df2= df1.groupby(['selectivity','benchmark']).agg({'throughput': 'max'})
    df2 = df2.rename(index={'bench1_base_variant': 'SPACE 1'})
    df2 = df2.rename(index={'bench2_base_variant_skipping': 'SPACE 2'})
    df2 = df2.rename(index={'bench3_3pass_streaming': 'SPACE 3'})
    df2 = df2.rename(index={'bench4_3pass_optimized_read_non_skipping_cub_pss': 'SPACE 4'})
    df2 = df2.rename(index={'bench5_3pass_optimized_read_skipping_partial_pss': 'SPACE 5'})
    df2 = df2.rename(index={'bench6_3pass_optimized_read_skipping_two_phase_pss': 'SPACE 6'})
    df2 = df2.rename(index={'bench7_3pass_optimized_read_skipping_cub_pss': 'SPACE 7'})
    df2 = df2.rename(index={'bench8_cub_flagged': 'cub::Flagged'})
    df2 = df2.rename(index={'bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss': 'SPACE 8'})

    #print(df2)

    if(multidim):
        p=sns.lineplot(ax=axes[pos][pos2],  x="selectivity",ci=None, y="throughput",hue='benchmark',  data=df2,
                         dashes=False, markers=True,style="benchmark")
    else:
        p = sns.lineplot(ax=axes[pos], x="selectivity", ci=None, y="throughput", hue='benchmark', data=df2,
                         dashes=False, markers=True, style="benchmark")
    if(log):
         p.set_xscale('logit')
    if(logy):
         p.set_yscale('log')

    p.set_xlabel("% selected data")
    p.set_ylabel("Throughput [GiB/s]")

    # ampere 1448,207 GiB/s
    # draw line for GPU bandwith
    cub_value_start = df2['throughput'].iloc[1]
    space_value_start = df2['throughput'].iloc[0]
    cub_value_end = df2.tail(2)
    cub_value_end = cub_value_end['throughput'].iloc[1]
    space_value_end = df2.tail(2)
    space_value_end = space_value_end['throughput'].iloc[0]
    speedup_start = space_value_start / cub_value_start
    speedup_end = space_value_end / cub_value_end
    print(speedup_start)
    print(speedup_end)
    # ampere 1448,207 GiB/s
    gpu_bandwith = 1448


    if (intro):
        axes[pos].hlines(xmin=0.01, xmax=0.99, y=gpu_bandwith, linewidth=2, color='grey', linestyles="dotted",
                         label="Bandwith: " + str("{:1d}".format(gpu_bandwith)) + "[GiB/s]")

        axes[pos].vlines( ymin=cub_value_start, ymax=space_value_start, x=0.0, linewidth=2, color='black',linestyles ="dashed",
                          label="Speedup 1%: " + str("{:1.2f}".format(speedup_start)))

        axes[pos].vlines(ymin=cub_value_end, ymax=space_value_end, x=0.97, linewidth=2, color='black',linestyles ="dashed",
                label="Speedup 97%: "+ str("{:1.2f}".format(speedup_end)))

    if (multidim):
        if(label1==False):
            axes[pos][pos2].yaxis.label.set_visible(False)
        else:
            axes[pos][pos2].yaxis.label.set_visible(True)

    if (label1 == False):
         axes[pos].yaxis.label.set_visible(False)
    else:
         axes[pos].yaxis.label.set_visible(True)




    #remove to restore legend
    if (multidim):
        axes[pos][pos2].get_legend().remove()
    else:
        axes[pos].legend(loc=1)
    #add to render legend within graph
    #axes[pos][pos2].legend(loc=1)
    #sns_plot.savefig(opath + label + ".png")
    #plt.savefig(opath + label + ".pdf", bbox_inches='tight')
    return axes



def draw_grid_renaming(path,  axes, pos,label1=True, draw_xlabel=True, multidim=False, pos2=0):

    df1 = pd.read_csv(path, sep=';')
    opath = "vis/"
    #print(df1)
    #df1 = df1.loc[df1['cluster_count'] != '32']
    #df1 = df1.loc[df1['benchmark'] != 'bench1_base_variant']
    #df1 = df1.loc[df1['benchmark'] != 'bench2_base_variant_skipping']
    #df1 = df1.loc[df1['benchmark'] != 'bench3_3pass_streaming']
    #df1 = df1.loc[df1['benchmark'] != 'bench4_3pass_optimized_read_non_skipping_cub_pss']
    #df1 = df1.loc[df1['benchmark'] != 'bench5_3pass_optimized_read_skipping_partial_pss']
    #df1 = df1.loc[df1['benchmark'] != 'bench6_3pass_optimized_read_skipping_two_phase_pss']
    #df1 = df1.loc[df1['benchmark'] != 'bench7_3pass_optimized_read_skipping_cub_pss']
    #df1 = df1.groupby(['selectivity','benchmark']).agg({'time_total': 'min'})

    df2= df1.groupby(['selectivity','benchmark']).agg({'throughput': 'max'})

    df2 = df2.rename(index={'bench1_base_variant': 'SPACE 1'})
    df2 = df2.rename(index={'bench2_base_variant_skipping': 'SPACE 2'})
    df2 = df2.rename(index={'bench3_3pass_streaming': 'SPACE 3'})
    df2 = df2.rename(index={'bench4_3pass_optimized_read_non_skipping_cub_pss': 'SPACE 4'})
    df2 = df2.rename(index={'bench5_3pass_optimized_read_skipping_partial_pss': 'SPACE 5'})
    df2 = df2.rename(index={'bench6_3pass_optimized_read_skipping_two_phase_pss': 'SPACE 6'})
    df2 = df2.rename(index={'bench7_3pass_optimized_read_skipping_cub_pss': 'SPACE 7'})
    df2 = df2.rename(index={'bench8_cub_flagged': 'cub::DeviceSelect::Flagged'})
    df2 = df2.rename(index={'bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss': 'SPACE 8'})



    #print(df2)

    if(multidim):
        p=sns.lineplot(ax=axes[pos][pos2],  x="selectivity",ci=None, y="throughput",hue='benchmark',  data=df2,
                         dashes=False, markers=True,style="benchmark")

    else:
        p = sns.lineplot(ax=axes[pos], x="selectivity", ci=None, y="throughput", hue='benchmark', data=df2,
                         dashes=False, markers=True, style="benchmark")

    p.set_xlabel("% selected data")
    p.set_ylabel("Throughput [GiB/s]")
    p.set_xlim(xmin=0, xmax=1)


    cub_value_start = df2['throughput'].iloc[1]
    space_value_start = df2['throughput'].iloc[0]
    cub_value_end = df2.tail(2)
    cub_value_end =cub_value_end['throughput'].iloc[1]
    space_value_end = df2.tail(2)
    space_value_end = space_value_end['throughput'].iloc[0]
    speedup_start = space_value_start/cub_value_start
    speedup_end = space_value_end/cub_value_end
    print(speedup_start)
    print(speedup_end)
    #ampere 1448,207 GiB/s
    #draw line for GPU bandwith
    #gpu_bandwith=1448
    #if multidim:
     #   axes[pos][pos2].hlines(xmin=0.01, xmax=0.99, y=gpu_bandwith, linewidth=2, color='grey', linestyles="dotted",
      #                   label="Bandwith: " + str("{:1d}".format(gpu_bandwith)) + "[GiB/s]")
    #else:
     #   axes[pos].hlines(xmin=0.01, xmax=0.99, y=gpu_bandwith, linewidth=2, color='grey', linestyles="dotted",
      #               label="Bandwith: " + str("{:1d}".format(gpu_bandwith))+"[GiB/s]")

    if(label1==False):
        axes[pos][pos2].yaxis.label.set_visible(False)
    if(draw_xlabel):
        axes[pos][pos2].xaxis.label.set_visible(True)
    else:
        axes[pos][pos2].xaxis.label.set_visible(False)
        axes[pos][pos2].xaxis.set_ticklabels([])


    #remove to restore legend
    axes[pos][pos2].get_legend().remove()
    #add to render legend within graph
    #axes[pos][pos2].legend(loc=1)
    #sns_plot.savefig(opath + label + ".png")
    #plt.savefig(opath + label + ".pdf", bbox_inches='tight')
    return axes

def draw_ampere_intro():

    pfade = [
        "data/2b/ampere/single_cluster_ampere_uint32_t.txt",
        "data/2b/ampere/multi_cluster_ampere_uint32_t.txt",
        "data/2b/ampere/uniform_ampere_uint32_t.txt"]


    sns.set_style("whitegrid")
    sns.color_palette("viridis")
    #define grid id subplots
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10 ,5))
    #fig.suptitle("Compaction SPACE compared to CUB")
    axes[0].set_title("single cluster")
    axes[1].set_title("multiple cluster")
    axes[2].set_title("uniform")

    axes= draw_grid(pfade[0], axes,0, intro=True)
    axes=draw_grid(pfade[1],  axes, 1,False,intro=True)
    axes=draw_grid(pfade[2],  axes, 2,False,intro=True)

    #axes[1].set_yticks([])
    #axes[2].set_yticks([])

    fig.savefig("vis/a100_intro" + ".png")
    plt.savefig("vis/a100_intro"+ ".pdf", bbox_inches='tight')
    #draw_cat(  pfad,label1=label,)
    #draw_bestvscub(pfad,label)

    plt.show()

def draw_lowsel():

    pfade = [
        "data/2b/quadro_lowsel/single_cluster__lowsel_quadro_uint32_t.txt",
        "data/2b/quadro_lowsel/multi_cluster__lowsel_quadro_uint32_t.txt",
        "data/2b/quadro_lowsel/uniform__lowsel_quadro_uint32_t.txt"]


    sns.set_style("whitegrid")
    sns.color_palette("viridis")
    #define grid id subplots
    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(10 ,5), sharey=True)
    #fig.suptitle("Compaction SPACE compared to CUB")
    axes[0].set_title("single cluster")
    axes[1].set_title("multiple cluster")
    axes[2].set_title("uniform")




    axes= draw_grid(pfade[0], axes,0,log=True)
    axes=draw_grid(pfade[1],  axes, 1,False,log=True )
    axes=draw_grid(pfade[2],  axes, 2,False,log=True)




    fig.savefig("vis/lowsel_quadro" + ".png")
    plt.savefig("vis/lowsel_quadro"+ ".pdf", bbox_inches='tight')


    plt.show()

def chunksizes():
    pfade =         ["data/2b/ampere/single_cluster_ampere_uint32_t.txt"  ]
    df1 = pd.read_csv(pfade[0], sep=';')
    # df1 = df1.loc[df1['cluster_count'] != '32']
   # df1 = df1.loc[df1['benchmark'] != 'bench1_base_variant']
    #df1 = df1.loc[df1['benchmark'] != 'bench2_base_variant_skipping']
   # df1 = df1.loc[df1['benchmark'] != 'bench3_3pass_streaming']
    #df1 = df1.loc[df1['benchmark'] != 'bench4_3pass_optimized_read_non_skipping_cub_pss']
    #df1 = df1.loc[df1['benchmark'] != 'bench5_3pass_optimized_read_skipping_partial_pss']
    #df1 = df1.loc[df1['benchmark'] != 'bench6_3pass_optimized_read_skipping_two_phase_pss']
    df1 = df1.loc[df1['benchmark'] != 'bench8_cub_flagged']
    # df1 = df1.loc[df1['benchmark'] != 'bench7_3pass_optimized_read_skipping_cub_pss']
    # df1 = df1.groupby(['selectivity','benchmark']).agg({'time_total': 'min'})


    df2 = df1.groupby(['selectivity', 'benchmark','chunk_length']).agg({'throughput': 'max'})
    df2 = df2.reset_index()
    #over 25% sel max 1159 min 1143
    #top 2 max 1842, min 1762

    print(df2.describe())
    for index, row in df2.iterrows():
        print(row['chunk_length'],row['benchmark'],row['throughput'],row['selectivity'] )
    #1034
    #print(df1)
    #974 min chunk value
    #1088 max value

def select_dt(df8):
    df8 = df8.loc[df8['benchmark'] != 'bench1_base_variant']
    df8 = df8.loc[df8['benchmark'] != 'bench2_base_variant_skipping']
    df8 = df8.loc[df8['benchmark'] != 'bench3_3pass_streaming']
    df8 = df8.loc[df8['benchmark'] != 'bench4_3pass_optimized_read_non_skipping_cub_pss']
    df8 = df8.loc[df8['benchmark'] != 'bench5_3pass_optimized_read_skipping_partial_pss']
    df8 = df8.loc[df8['benchmark'] != 'bench6_3pass_optimized_read_skipping_two_phase_pss']
    df8 = df8.loc[df8['benchmark'] != 'bench7_3pass_optimized_read_skipping_cub_pss']

    return df8

def draw_datatypes():
    pfade = [
        "data/2b/ampere/single_cluster_ampere_double.txt",
        "data/2b/ampere/single_cluster_ampere_float.txt",
        "data/2b/ampere/single_cluster_ampere_uint8_t.txt"]


    df8 = pd.read_csv(pfade[0], sep=';')
    df4 = pd.read_csv(pfade[1], sep=';')
    df1 = pd.read_csv(pfade[2], sep=';')

    df8=select_dt(df8)
    df4 = select_dt(df4)
    df1 = select_dt(df1)

    df1 = df1.groupby(['selectivity', 'benchmark']).agg({'throughput': 'max'})
    df1 = df1.rename(index={'bench8_cub_flagged': 'cub::DeviceSelect::Flagged uint8_t'})
    df1 = df1.rename(index={'bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss': 'SPACE 8 uint8_t'})
    df4 = df4.groupby(['selectivity', 'benchmark']).agg({'throughput': 'max'})
    df4 = df4.rename(index={'bench8_cub_flagged': 'cub::DeviceSelect::Flagged float'})
    df4 = df4.rename(index={'bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss': 'SPACE 8 float'})
    df8 = df8.groupby(['selectivity', 'benchmark']).agg({'throughput': 'max'})
    df8 = df8.rename(index={'bench8_cub_flagged': 'cub::DeviceSelect::Flagged double'})
    df8 = df8.rename(index={'bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss': 'SPACE 8 double'})




    frames = [df1,df4,df8]
    df_all = pd.concat(frames)

    p = sns.lineplot( x="selectivity", ci=None, y="throughput", hue='benchmark', data=df_all,
                     dashes=False, markers=True, style="benchmark")
    p.set_xlabel("% selected data")
    p.set_ylabel("Throughput [GiB/s]")
    p.set_xlim(xmin=0, xmax=1)

    plt.savefig("vis/dtypes" + ".png")
    plt.savefig("vis/dtypes" + ".pdf", bbox_inches='tight')
    plt.show()


def draw_4x3():
    # input of all files
    pfade = [
        "data/2b/ampere/single_cluster_ampere_uint32_t.txt",
        "data/2b/ampere/multi_cluster_ampere_uint32_t.txt",
        "data/2b/ampere/uniform_ampere_uint32_t.txt",
        "data/2b/quadro/single_cluster__uint32_t.txt",
        "data/2b/quadro/multi_cluster__uint32_t.txt",
        "data/2b/quadro/uniform__uint32_t.txt",
        "data/2b/vrpc/single_cluster_quadro_uint32_t.txt",
        "data/2b/vrpc/multi_cluster_quadro_uint32_t.txt",
        "data/2b/vrpc/uniform_quadro_uint32_t.txt",
        "data/2b/3080/single_cluster_3080_uint32_t.txt",
        "data/2b/3080/multi_cluster_3080_uint32_t.txt",
        "data/2b/3080/uniform_3080_uint32_t.txt"
    ]

    sns.set_style("whitegrid")
    sns.color_palette("viridis")
    # define grid id subplots
    fig, axes = plt.subplots(5, 3, sharex=False, figsize=(10, 15))
    #overall title of grid
    #fig.suptitle("Best performing SPACE Compaction compared to CUB")
    axes[0][0].set_title("single cluster A100")
    axes[0][1].set_title("multiple cluster A100")
    axes[0][2].set_title("uniform A100")
    axes[1][0].set_title("single cluster RTX 8000")
    axes[1][1].set_title("multiple cluster RTX 8000")
    axes[1][2].set_title("uniform RTX 8000")
    axes[2][0].set_title("single cluster 1070 TI")
    axes[2][1].set_title("multiple cluster 1070 TI")
    axes[2][2].set_title("uniform 1070 TI")
    axes[3][0].set_title("single cluster 3080")
    axes[3][1].set_title("multiple cluster 3080")
    axes[3][2].set_title("uniform 3080")



    draw_grid_renaming(pfade[0], axes, 0, pos2=0, multidim=True,draw_xlabel=False )
    draw_grid_renaming(pfade[1], axes, 0, pos2=1, multidim=True, label1=False,draw_xlabel=False )
    draw_grid_renaming(pfade[2], axes, 0, pos2=2, multidim=True, label1=False,draw_xlabel=False )
    draw_grid_renaming(pfade[3], axes, 1, pos2=0, multidim=True,draw_xlabel=False )
    draw_grid_renaming(pfade[4], axes, 1, pos2=1, multidim=True, label1=False,draw_xlabel=False)
    draw_grid_renaming(pfade[5], axes, 1, pos2=2, multidim=True, label1=False,draw_xlabel=False)
    draw_grid_renaming(pfade[6], axes, 2, pos2=0, multidim=True,draw_xlabel=False)
    draw_grid_renaming(pfade[7], axes, 2, pos2=1, multidim=True, label1=False,draw_xlabel=False)
    draw_grid_renaming(pfade[8], axes, 2, pos2=2, multidim=True, label1=False,draw_xlabel=False)
    draw_grid_renaming(pfade[9], axes, 3, pos2=0, multidim=True)
    draw_grid_renaming(pfade[10], axes, 3, pos2=1, multidim=True, label1=False)
    draw_grid_renaming(pfade[11], axes, 3, pos2=2, multidim=True, label1=False)

    # Clear bottom-right ax
    bottom_right_ax = axes[4][2]
    bottom_right_ax.clear()  # clears the random data I plotted previously
    bottom_right_ax.set_axis_off()  # removes the XY axes
    bottom_right_ax = axes[4][1]
    bottom_right_ax.clear()  # clears the random data I plotted previously
    bottom_right_ax.set_axis_off()  # removes the XY axes
    bottom_right_ax = axes[4][0]
    bottom_right_ax.clear()  # clears the random data I plotted previously
    bottom_right_ax.set_axis_off()  # removes the XY axes

    #get legend from frist graph and show in gridspace 4,1
    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[4][1].legend(handles, labels)

    # axes[1].set_yticks([])
    # axes[2].set_yticks([])

    fig.savefig("vis/4x3all" + ".png")
    plt.savefig("vis/4x3all" + ".pdf", bbox_inches='tight')
    # draw_cat(  pfad,label1=label,)
    # draw_bestvscub(pfad,label)

    plt.show()

def draw_3x3():
    #input of all files
    pfade = [
        "data/2b/ampere/single_cluster_ampere_uint32_t.txt",
        "data/2b/ampere/multi_cluster_ampere_uint32_t.txt",
        "data/2b/ampere/uniform_ampere_uint32_t.txt",
        "data/2b/quadro/single_cluster__uint32_t.txt",
        "data/2b/quadro/multi_cluster__uint32_t.txt",
        "data/2b/quadro/uniform__uint32_t.txt",
        "data/2b/vrpc/single_cluster_quadro_uint32_t.txt",
        "data/2b/vrpc/multi_cluster_quadro_uint32_t.txt",
        "data/2b/vrpc/uniform_quadro_uint32_t.txt"
    ]


    sns.set_style("whitegrid")
    sns.color_palette("viridis")
    #define grid id subplots
    fig, axes = plt.subplots(4, 3,  figsize=(10 ,15))
    #fig.suptitle("Best performing SPACE Compaction compared to CUB")
    axes[0][0].set_title("single cluster A100")
    axes[0][1].set_title("multiple cluster A100")
    axes[0][2].set_title("uniform A100")
    axes[1][0].set_title("single cluster RTX 8000")
    axes[1][1].set_title("multiple cluster RTX 8000")
    axes[1][2].set_title("uniform RTX 8000")
    axes[2][0].set_title("single cluster 1070 TI")
    axes[2][1].set_title("multiple cluster 1070 TI")
    axes[2][2].set_title("uniform 1070 TI")


    draw_grid_renaming(pfade[0], axes, 0, pos2=0,multidim=True, draw_xlabel=False)
    draw_grid_renaming(pfade[1], axes, 0, pos2=1, multidim=True, label1=False , draw_xlabel=False)
    draw_grid_renaming(pfade[2], axes, 0, pos2=2, multidim=True, label1=False, draw_xlabel=False)
    draw_grid_renaming(pfade[3], axes, 1, pos2=0, multidim=True, draw_xlabel=False)
    draw_grid_renaming(pfade[4], axes, 1, pos2=1, multidim=True, label1=False, draw_xlabel=False)
    draw_grid_renaming(pfade[5], axes, 1, pos2=2, multidim=True, label1=False, draw_xlabel=False)
    draw_grid_renaming(pfade[6], axes, 2, pos2=0, multidim=True)
    draw_grid_renaming(pfade[7], axes, 2, pos2=1, multidim=True, label1=False)
    draw_grid_renaming(pfade[8], axes, 2, pos2=2, multidim=True, label1=False)

    # Clear bottom-right ax
    bottom_right_ax = axes[3][2]
    bottom_right_ax.clear()  # clears the random data I plotted previously
    bottom_right_ax.set_axis_off()  # removes the XY axes
    bottom_right_ax = axes[3][1]
    bottom_right_ax.clear()  # clears the random data I plotted previously
    bottom_right_ax.set_axis_off()  # removes the XY axes
    bottom_right_ax = axes[3][0]
    bottom_right_ax.clear()  # clears the random data I plotted previously
    bottom_right_ax.set_axis_off()  # removes the XY axes

    handles, labels = axes[0][0].get_legend_handles_labels()
    axes[3][1].legend(handles, labels)




    #axes[1].set_yticks([])
    #axes[2].set_yticks([])

    fig.savefig("vis/3x3all" + ".png")
    plt.savefig("vis/3x3all"+ ".pdf", bbox_inches='tight')
    #draw_cat(  pfad,label1=label,)
    #draw_bestvscub(pfad,label)

    plt.show()

if __name__ == '__main__':
    pfad = sys.argv[1]
    label = sys.argv[2]




    sns.set_style("whitegrid")
    sns.color_palette("viridis")


    #chunksizes()
    #draw_datatypes()

    #draw 3x3 grid 1070ti, a100, rtx8000
    #draw_3x3()

    #like 3x3 + 3080 GPU
    draw_4x3()

    #ampere intro graphic 3x1 plot
    #draw_ampere_intro()
    #draw_lowsel()

    #draw_cat(  pfad,label1=label,)
    #draw_bestvscub(pfad,label)

    plt.show()
    gc.collect()


