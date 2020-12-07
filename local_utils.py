"""MIT License

Copyright (c) 2020 Jordan Berg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import pandas as pd
from sklearn import preprocessing
from math import sqrt
from statistics import mean, stdev
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import GEOparse
import xpressplot as xp
sns.set(font='arial')
_jakes_cmap = sns.diverging_palette(
    212, 61, s=99, l=77, sep=1, n=16, center='dark') #Custom aesthetics
__path__  = os.getcwd()

def cohen_d(comp, base):
    return (mean(comp) - mean(base)) / (sqrt((stdev(comp) ** 2 + stdev(base) ** 2) / 2))

def eval_cohens(e):
    """Cohen's d effect size scale:
    Very small   0.01
    Small        0.20
    Medium       0.50
    Large        0.80
    Very large   1.20
    Huge         2.0

    Source:
    - Cohen, Jacob (1988). Statistical Power Analysis for the Behavioral Sciences.
        Routledge. ISBN 978-1-134-74270-7.
    - Sawilowsky, S (2009). "New effect size rules of thumb".
        Journal of Modern Applied Statistical Methods. 8 (2): 467–474.
        doi:10.22237/jmasm/1257035100.
    """

    status = ""
    if abs(e) < 0.01:
        status = "Negligible"
    elif abs(e) >= 0.01 and abs(e) < 0.2:
        status = "Very small"
    elif abs(e) >= 0.2 and abs(e) < 0.5:
        status = "Small"
    elif abs(e) >= 0.5 and abs(e) < 0.8:
        status = "Medium"
    elif abs(e) >= 0.8 and abs(e) < 1.2:
        status = "Large"
    elif abs(e) >= 1.2 and abs(e) < 2.0:
        status = "Very large"
    else:
        status = "Huge"
    return status

def eval_gene(
        gene,
        data,
        info,
        palette,
        name):

    if gene not in data.index.tolist():
        print(gene + ' not found in dataset ' + name + '. Skipping...')
    else:
        print('---------------------------------------------------------------')
        print(gene + ' in dataset ' + name)
        print('---------------------------------------------------------------')

        info.columns = [0,1]
        data_scaled, data_labeled = xp.prep_data(data, info)
        xp.gene_overview(
            data_labeled,
            info,
            gene_name=gene,
            palette=palette,
            grid=True,
            whitegrid=True)
        plt.show()

        print('Effect sizes:')
        data_effect = data.copy()
        data_effect = data_effect.loc[gene]
        labels = list(palette.keys())
        evaluated = []
        for l in labels:
            for ll in labels:
                if l != ll and {l, ll} not in evaluated:
                    base_list = []
                    comp_list = []
                    for i, r in info.iterrows():
                        if r[1] == l:
                            base_list.append(r[0])
                        elif r[1] == ll:
                            comp_list.append(r[0])
                        else:
                            pass
                    base = data_effect[base_list]
                    comp = data_effect[comp_list]
                    e = cohen_d(comp, base)
                    status = eval_cohens(e)
                    print(
                        l + ' vs ' + ll + ':\t'
                        + str(round(e, 2)) + "\t" +
                        "(" + status + ")")
                    evaluated.append({l, ll})

    print('\n\n')

def make_heatmap(
        gene_list,
        data,
        info,
        palette,
        name,
        constrain_values=True,
        _font_scale=0.8,
        _center=0,
        _metric='euclidean',
        _method='centroid',
        _xticklabels=False,
        _yticklabels=True,
        _linewidths=0,
        _linecolor='#DCDCDC',
        _col_cluster=False,
        _row_cluster=True,
        _fig_width=15):

    print('---------------------------------------------------------------')
    print('Dataset ' + name)
    print('---------------------------------------------------------------')

    updated_genes = []
    for gene in gene_list:
        if gene not in data.index.tolist():
            print(gene + ' not found in ' + name + ' dataset. Removing from the provided list.')
        else:
            updated_genes.append(gene)

    if len(updated_genes) <= 0:
        print('No genes to plot for dataset ' + name + '. Skipping...')
    else:
        data_genes = data.copy()
        data_genes = data_genes.loc[updated_genes]

        if constrain_values == True:
            data_genes[data_genes > 5] = 5
            data_genes[data_genes < -5] = -5

        # Prepare sample color map
        info.columns = [0,1]
        info = info.T
        info.columns = info.iloc[0]
        info = info.reindex(info.index.drop(0))
        info = info.rename({1: 'samples'})
        labels = info.iloc[0]
        _color_map = labels.map(palette)

        sns.set(font_scale = _font_scale)
        ax = sns.clustermap(
            data_genes,
            cmap=_jakes_cmap,
            center=_center,
            metric=_metric,
            method=_method,
            xticklabels=_xticklabels,
            yticklabels=_yticklabels,
            linewidths=_linewidths,
            linecolor=_linecolor,
            col_cluster=_col_cluster,
            row_cluster=_row_cluster,
            col_colors=_color_map,
            figsize=(_fig_width, len(updated_genes) / 1.5))

        g = lambda m,c: plt.plot([],[],marker='o', color=c, ls="none")[0]
        handles_g = [
            g("s", list(palette.values())[i]) for i in range(len(list(palette.values())))]
        plt.legend(
            handles_g,
            list(palette.keys()),
            bbox_to_anchor=(0, -7),
            loc=3,
            borderaxespad=0.,
            title='Samples')
        plt.show()

    print('\n\n')

def prep_tcga():
    # Import TCGA/GTEx Unity Data
    data_dir = os.path.join(__path__, '_data', 'tcga_unity_data') + os.path.sep
    file_list = ['colon-rsem-fpkm-gtex.zip',
                'coad-rsem-fpkm-tcga.zip',
                'coad-rsem-fpkm-tcga-t.zip']
    file_list = [str(data_dir) + str(x) for x in file_list]

    #Initialize
    df_list = []
    meta = pd.DataFrame(columns = [0,1])
    previous = 0
    label_number = 0
    drop_extra='Entrez_Gene_Id'
    label_list = [
        'GTEX_Normal',
        'TCGA_Normal',
        'TCGA_Tumor']

    #Import
    for x in range(len(file_list)):
        #Get and clean data
        data = pd.read_csv(
            str(file_list[x]),
            index_col = 0,
            sep = '\t')
        data.index.name = None

        #Drop extra column if needed
        if drop_extra != None:
            data = data.drop(labels=['Entrez_Gene_Id'], axis=1)

        #Add data to data list
        df_list.append(data)
        print('Size of ' + str(file_list[x]) + ': ' + str(data.shape))

        #Get metadata for current data matrix
        data_L = data.columns.tolist()
        for x in range(len(data_L)):
            meta.loc[previous + x] = [data_L[x], label_list[label_number]]

        #Update iterators
        previous = previous + len(data_L)
        label_number += 1

    #Join dataframes
    data = pd.concat(df_list, axis=1)
    print('Size of final dataframe: ' + str(data.shape))
    print('Size of final metadata table: ' + str(meta.shape))
    data.to_csv(
        os.path.join(__path__, '_data', 'tcga_unity_data.txt'),
        sep='\t'
    )
    meta.to_csv(
        os.path.join(__path__, '_data', 'tcga_unity_info.txt'),
        sep='\t'
    )

def init_tcga(
        file=os.path.join(__path__, "_data", "tcga_unity_data.txt"),
        metadata=os.path.join(__path__, "_data", "tcga_unity_info.txt")
        ):

    if not os.path.exists(file) or not os.path.exists(metadata):
        print('Generating TCGA data for analysis...')
        prep_tcga()

    data = pd.read_csv(
        file,
        sep='\t',
        index_col=0)
    info = pd.read_csv(
        metadata,
        sep='\t',
        index_col=0)
    scaled_data = data.copy()
    scaled_data[scaled_data.columns] = preprocessing.scale(scaled_data[scaled_data.columns], axis=1)
    palette = {
        'GTEX_Normal': '#005f42',
        'TCGA_Normal': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
        'TCGA_Tumor': (0.5725490196078431, 0.5843137254901961, 0.5686274509803921)}

    return scaled_data, data, info, palette

def get_data(
        file,
        metadata,
        sep=','):

    data = pd.read_csv(
        file,
        sep=sep,
        index_col=0)
    info = xp.get_info(
        metadata,
        delimiter=',')
    scaled_data = data.copy()
    scaled_data[scaled_data.columns] = preprocessing.scale(
        scaled_data[scaled_data.columns],
        axis=1)

    return scaled_data, data, info

def prep_GSE8671():

    #Get data
    df_GSE8671 = xp.get_df(
        os.path.join(__path__, '_data', 'GSE8671', 'GSE8671_rma_normalized.zip'),
        delimiter=',') #RMA normalized with Alt Analyze
    info_GSE8671 = xp.get_info(
        os.path.join(__path__, '_data', 'GSE8671', 'sample_info_gse8671.csv'),
        delimiter=',')
    df_GSE8671_c = xp.keep_labels(
        df_GSE8671,
        info_GSE8671,
        label_list=['Normal','Adenoma'])
    df_GSE8671_clean = xp.clean_df(df_GSE8671_c)

    #Collapse multi-mapping probes
    df_GSE8671_collapsed = xp.probe_collapse(
        df_GSE8671_clean,
        os.path.join(__path__, '_data', 'GPL570.zip'))
    df_GSE8671_collapsed.to_csv(
        os.path.join(__path__, '_data', 'collapsed_GSE8671.csv'),
        sep=',')

def init_GSE8671(
        file=os.path.join(__path__, '_data', 'collapsed_GSE8671.csv'),
        metadata=os.path.join(__path__, '_data', 'GSE8671', 'sample_info_gse8671.csv')):

    if not os.path.exists(file) or not os.path.exists(metadata):
        print('Generating GSE8671 data for analysis...')
        prep_GSE8671()

    scaled_data, data, info = get_data(
        file=file,
        metadata=metadata)
    palette = {
        'Adenoma': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
        'Normal': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)}

    return scaled_data, data, info, palette

def get_geo(
        geo_id,
        output_info=False,
        output_path=os.path.join(__path__, '_data', 'GSE20916')):

    # Get data
    gse = GEOparse.get_GEO(
        geo=str(geo_id).upper(),
        destdir=output_path) # Import GSE dataset
    data = gse.pivot_samples('VALUE')
    data = xp.clean_df(data)

    # Get metadata
    # Write data to output file
    if output_info != False:
        with open(str(geo_id).upper() + '.txt', 'w+') as f: # Save all information as text file for reference
            for gsm_name, gsm in gse.gsms.items():
                f.write(gsm_name + '\n')
                for key, value in gsm.metadata.items():
                    f.write(" - %s : %s" % (key, ", ".join(value)) + '\n')

    # Populate metadata with sample ids and names
    metadata = pd.DataFrame(columns=['gsm', 'title']) # Create dataframe
    gsm_list, title_list, data_processing_list = [], [], []
    for gsm_name, gsm in gse.gsms.items():
        for key, value in gsm.metadata.items():
            if key == 'title':
                title_list.append(''.join(value))
            if key == 'geo_accession':
                gsm_list.append(''.join(value))
            if key == 'data_processing':
                data_processing_list.append(''.join(value))
    metadata['gsm'], metadata['title'] = gsm_list, title_list
    metadata.columns = range(metadata.shape[1])

    # Output processing style
    print('Data processing summary:\n' + str(set(data_processing_list))) # To determine if all samples have undergone the sample data processing

    # Clean data
    data.columns.name = None
    data.index.name = None

    # Clean metadata
    metadata[1] = metadata[1].apply(
        lambda x: x[0: (re.search("\d", x).start()) - 1])

    return data, metadata

def prep_GSE20916():

    #Get data
    df_GSE20916 = pd.read_csv(
        os.path.join(__path__ + "_data", "GSE20916", "GSE20916_normalized.zip"),
        index_col=0)
    info_GSE20916 = pd.read_csv(
        os.path.join(__path__ + "_data", "GSE20916", "sample_info_gse20916.csv"))
    info_GSE20916.columns = [0,1]
    info_GSE20916[1] = info_GSE20916[1].str.capitalize() #Make sample types look nice
    info_GSE20916 = info_GSE20916.replace('Normal_colon', 'Normal')
    df_GSE20916_c = xp.keep_labels(
        df_GSE20916,
        info_GSE20916,
        label_list=['Normal','Adenoma','Adenocarcinoma'])
    df_GSE20916_clean = xp.clean_df(df_GSE20916_c)

    #Collapse multi-mapping probes
    df_GSE20916_collapsed = xp.probe_collapse(
        df_GSE20916_clean,
        os.path.join(__path__, "_data", "GPL570.zip"))
    df_GSE20916_collapsed.to_csv(
        os.path.join(__path__ + "_data", "collapsed_GSE20916.txt",
        sep='\t'))

def init_GSE20916(
        file=os.path.join(__path__, "_data", "collapsed_GSE20916.txt"),
        metadata=os.path.join(__path__, "_data", "GSE20916", "sample_info_gse20916.csv")):

    if not os.path.exists(file) or not os.path.exists(metadata):
        print('Generating GSE8671 data for analysis...')
        prep_GSE8671()

    scaled_data, data, info = get_data(
        file=file,
        metadata=metadata,
        sep='\t')
    palette = {
        'adenocarcinoma': (0.5725490196078431, 0.5843137254901961, 0.5686274509803921),
        'adenoma': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
        'normal_colon': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275)}

    return scaled_data, data, info, palette

def sort_data(data, info, names):

    info_sorted = info.copy()
    info_sorted = info_sorted.loc[info_sorted[1].isin(names)]
    info_sorted = info_sorted.sort_values([1], ascending=False)
    info_sorted_list = info_sorted[0].tolist()
    data_sorted = data[info_sorted_list]

    return data_sorted
