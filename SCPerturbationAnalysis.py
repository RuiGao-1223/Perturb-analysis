import scanpy as sc
import pandas as pd
import math
import numpy as np
import scipy
from scipy import stats
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import warnings
warnings.filterwarnings("ignore")
from statsmodels.stats.multitest import multipletests
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import mplcursors
# import matplotlib.cm as cm
# from multiprocessing import Process,Manager

class SCPerturbationAnalysis:


    '''
    Basic info statistics for perturbation single cell anndata:
    

    PARAMETERS:
    file_path: tha path of .h5ad file

    
    ATTRIBUTE:

    gene_lfc: 
        Log Fold Change of each gene for every perturbation. 
        DataFrame: Perturbation*Gene

    gene_pvalue: 
        Mutiple testing correleted p-value of each gene for every perturbation.
        DataFrame: Perturbation*Gene

    total_mRNA_info:
        total mRNA info of before and after the peturbation.
        DataFrame: Perturbation*Features
        # Features:
        #   total mRNA: total mRNA after perturbation
        #   ratio: (total mRNA of after perturbation)/(total mRNA of no_targeting)
        # 	logFC: Log Fold Change of total mRNA compared with no_targeting
        # 	p_value: p-value for each perturbation

    knoked gene info:
        knocked gene expression level and related value(compared to total mRNA change).
        DataFrame: Perturbation*Features

    directly_related_gene:
        downstream genes inof for every perturbation, including two parts:
            downstream_counts: 
                For each perturbation, the downstream gene counts
                DataFrame: Perturbation*1(counts)
            downstream_genelist:
                For each perturbation, the downstream gene list
                DataFrame: Perturbation*genes
    
    FUNCTION:
    get_all_info(gene):
        output all above info for specific gene


    '''

    def __init__(self, file_path):
        self.__anndata = sc.read_h5ad(file_path)
        self.__expression = pd.DataFrame(self.__anndata.X, columns=self.__anndata.var.index, index=self.__anndata.obs.index).T
        self.__perturb_fliter,self.__perturbation = self._fliter_by_cell_num()
        # self.__perturbation=list(set(self.__anndata.obs['gene']))
        self.__gene_exp = self._calculate_gene_exp()
        self.__lfc = self._calculate_lfc()
        self.__pvalue = self._calculate_gene_pvalue()
        self.__total_mRNA = self._calculate_total_mRNA()
        self.__knock_gene_df = self._calculate_knocked_gene_info()
        self.__downstream_counts, self.__downstream_genelist=self._find_downstream_gene()
        self.__uptream_counts, self.__upstream_genelist=self._find_upstream_gene()

    def _fliter_by_cell_num(self):
        ### generate cell number distribution
        perturbation=list(set(self.__anndata.obs['gene']))
        cell_num=[]
        for gene in perturbation:
            expriment=self.__expression[self.__anndata.obs[self.__anndata.obs['gene']==gene].index]
            cell_num.append(expriment.shape[1])
        cell_num=pd.DataFrame(cell_num,index=perturbation,columns=['cell_num'])

        total_rna_std_log = [np.std(np.log2(self.__anndata.obs[self.__anndata.obs['gene'] == gene]['UMI_count'])) for gene in perturbation]
        total_rna_log = [np.log2(self.__anndata.obs[self.__anndata.obs['gene'] == gene]['UMI_count']).mean() for gene in perturbation]
        total_rna_std=np.exp(total_rna_std_log)
        total_rna = np.exp(total_rna_log)
        perturb_fliter=pd.DataFrame(zip(total_rna,total_rna_std,total_rna_std/total_rna),index=perturbation,columns=['total_rna','total_rna_std','total_rna_cv'])
        perturb_fliter['cell_number']=cell_num['cell_num']

        ### update perturbation list after fliter with cell_number
        pertubation_flited=list(perturb_fliter[perturb_fliter['cell_number']>15].index)
        return perturb_fliter,pertubation_flited

    def _calculate_gene_exp(self):
        gene_exp = pd.DataFrame()
        for gene in self.__perturbation:
            cells = self.__anndata.obs[self.__anndata.obs['gene'] == gene]
            expression_filtered = pd.DataFrame(self.__expression[cells.index])
            gene_exp = gene_exp.append(expression_filtered.mean(axis=1), ignore_index=True)
        gene_exp.index = self.__perturbation
        return gene_exp

    def _calculate_lfc(self):
        lfc = self.__gene_exp / self.__gene_exp.loc['non-targeting']
        lfc = np.log2(lfc.replace([np.inf, -np.inf], [1000, -1000]))
        rename_dict = dict(zip(self.__anndata.var.index, self.__anndata.var['gene_name']))
        rename_dict['ENSG00000284024'] = 'MSANTD7'
        lfc.rename(columns=rename_dict, inplace=True)
        return lfc

    def _calculate_gene_pvalue(self):
        non_targeting = self.__expression[self.__anndata.obs[self.__anndata.obs['gene'] == 'non-targeting'].index]

        # Function to calculate p-value for a gene
        def ranksums_parallel(gene):
            multitest = False
            cells = self.__anndata.obs[self.__anndata.obs['gene'] == gene]
            expression_filtered = self.__expression[cells.index]
            p_values = expression_filtered.apply(lambda row: ranksums(row, non_targeting.loc[row.name])[1], axis=1)
            if multitest:
                p_values = multipletests(p_values, method='fdr_bh')[1]
            return p_values

        # Use Parallel to calculate p-values for all genes in parallel
        num_cores = -1  # Use all available cores, adjust as needed
        with Parallel(n_jobs=num_cores) as parallel:
            results = parallel(delayed(ranksums_parallel)(gene) for gene in self.__perturbation)

        gene_pvalue = pd.DataFrame({gene: p_values for gene, p_values in zip(self.__perturbation, results)})
        rename_dict = dict(zip(self.__anndata.var.index, self.__anndata.var['gene_name']))
        rename_dict['ENSG00000284024'] = 'MSANTD7'
        gene_pvalue.rename(index=rename_dict, inplace=True)
        gene_pvalue = gene_pvalue.T
        return gene_pvalue

    def _calculate_total_mRNA(self):
        control = self.__anndata.obs[self.__anndata.obs['gene'] == 'non-targeting']
        total_rna_log = [np.log2(self.__anndata.obs[self.__anndata.obs['gene'] == gene]['UMI_count']).mean() for gene in self.__perturbation]
        p_value = [scipy.stats.ttest_ind(np.log2(control['UMI_count']), np.log2(self.__anndata.obs[self.__anndata.obs['gene'] == gene]['UMI_count']))[1] for gene in self.__perturbation]
        adjusted_pvalue=multipletests(p_value,method='fdr_bh')[1]
        total_rna = np.exp(total_rna_log)
        total_mRNA = pd.DataFrame(total_rna, index=self.__perturbation, columns=['total_mRNA'])
        total_mRNA['ratio'] = total_mRNA / total_mRNA.loc['non-targeting', 'total_mRNA']
        total_mRNA['logFC'] = np.log2(total_mRNA['ratio'])
        total_mRNA['p_value'] = adjusted_pvalue
        return total_mRNA

    def _calculate_knocked_gene_info(self):
        gene_name_ary = []
        lfc_ary = []
        gene_ary_no_knock = []
        gene_ary_knocked = []
        rename_dict = dict(zip(self.__anndata.var.index, self.__anndata.var['gene_name']))
        rename_dict['ENSG00000284024'] = 'MSANTD7'
        self.__gene_exp.rename(columns=rename_dict, inplace=True)
        for knock_gene in self.__lfc.index:
            if knock_gene in self.__lfc.columns:
                gene_name_ary.append(knock_gene)
                lfc_ary.append(self.__lfc.loc[knock_gene, knock_gene])
                gene_ary_knocked.append(self.__gene_exp.loc[knock_gene, knock_gene])
                gene_ary_no_knock.append(self.__gene_exp.loc['non-targeting', knock_gene])
        knock_gene_df = pd.DataFrame(zip(gene_name_ary, lfc_ary, gene_ary_no_knock, gene_ary_knocked), columns=['gene_name', 'log FC', 'original', 'knocked'])
        knock_gene_df.set_index("gene_name", inplace=True)
        knock_gene_df['variation']=knock_gene_df['knocked']-knock_gene_df['original']
        ### 计算被敲除基因的相对值
        filtered = self.__total_mRNA[self.__total_mRNA.index.isin(knock_gene_df.index)]
        knock_gene_df['original_relative'] = knock_gene_df['original'] / self.__total_mRNA.loc['non-targeting', 'total_mRNA']
        knock_gene_df['knocked_relative'] = knock_gene_df['knocked'] / filtered['total_mRNA']
        knock_gene_df['variation_relative'] = knock_gene_df['knocked_relative'] - knock_gene_df['original_relative']
        return knock_gene_df
    
    def _find_downstream_gene(self):
        # Calculate the absolute difference between knocked and non-targeting for all pairs (i, j)
        # Pre-calculate the threshold for all rows
        non_targeting = self.__gene_exp.loc['non-targeting']
        thresholds = pd.DataFrame(np.percentile(np.abs(self.__gene_exp - non_targeting), 80, axis=1),index=self.__lfc.index)
        genelist = []

        for i in self.__lfc.index:
            abs_diff_non_targeting = np.abs(non_targeting - self.__gene_exp.loc[i])
            data=np.full((len(abs_diff_non_targeting),1),thresholds.loc[i])
            diff=pd.DataFrame(data,index=abs_diff_non_targeting.index)
            #
            mask = ((np.abs(self.__lfc.loc[i]) > 1.5).values)&(abs_diff_non_targeting > diff.iloc[:,0]) & ((self.__pvalue.loc[i].values < 0.05))
            gene_list = self.__lfc.columns[mask].tolist()
            genelist.append(gene_list)
       
        downstream_gene_list={key: values for key,values in zip(self.__lfc.index, genelist)}
        downstream_gene_list={key: [item for item in value if item!=key] for key,value in downstream_gene_list.items()}
        downstream_num=[]
        for key,value in downstream_gene_list.items():
            downstream_num.append(len(value))
        downstream_gene_counts=pd.DataFrame(downstream_num,index=self.__lfc.index,columns=['counts'])
        return downstream_gene_counts, downstream_gene_list
    
    def _find_upstream_gene(self):
        upstream_dict = {element: [] for element in self.__lfc.columns}
        for key in self.__downstream_genelist:
            for element in self.__lfc.columns:
                if element in self.__downstream_genelist[key]:
                    upstream_dict[element].append(key)
        upstream_num=[]
        for key,value in upstream_dict.items():
            upstream_num.append(len(value))
        upstream_num=pd.DataFrame(upstream_num,index=self.__lfc.columns,columns=['counts'])
        return upstream_num, upstream_dict
        
    @property
    def gene_exp(self):
        gene_exp=self.__gene_exp
        gene_exp.columns=self.__lfc.columns
        return gene_exp
    
    @property
    def perturb_fliter(self):
        return self.__perturb_fliter

    @property
    def gene_lfc(self):
        return self.__lfc

    @property
    def gene_pvalue(self):
        return self.__pvalue

    @property
    def total_mRNA_info(self):
        return self.__total_mRNA

    @property
    def knocked_gene_info(self):
        return self.__knock_gene_df

    @property
    def downstream_gene_counts(self):
        return self.__downstream_counts
    
    @property
    def downstream_gene_list(self):
        return self.__downstream_genelist   
    
    @property
    def uptream_gene_counts(self):
        return self.__uptream_counts
    
    @property
    def upstream_gene_list(self):
        return self.__upstream_genelist  
    
    def get_all_info_for_perturbation(self,gene):
        output_df1=pd.DataFrame(zip(self.__gene_exp.loc[gene].T, self.__gene_exp.loc['non-targeting'].T, self.__lfc.loc[gene].T, self.__pvalue.loc[gene]),columns=['gene_exp','gene_exp_control','lfc','gene_pvalue'], index=self.__lfc.columns).T
        print(f"knocked gene is {gene}, and the basic information is shown as below: ")
        print(output_df1[gene].to_string())
        print()
        print(f"cell number of this perturbation:{self.__perturb_fliter.loc[gene,'cell_number']}")
        print()
        print(f"total_mRNA info:")
        print(self.__total_mRNA.loc[gene])
        print()
        print(f"knocked gene info:")
        print(self.__knock_gene_df.loc[gene])
        print()
        print()
        print(f"downstream genes counts is: {self.__downstream_counts.loc[gene,'counts']}")
        print(f"downstream genes:")
        print(self.__downstream_genelist[gene])
        print()
        print(f"upstream genes counts is: {self.__uptream_counts.loc[gene,'counts']}")
        print(f"upstream genes:")
        print(self.__upstream_genelist[gene])
        print()

        # self.scatter_vocano_for_gene(gene,True)

        
        ### 对于要观测的target gene，观测其下游基因的表达分布及该perturbation在分布中的位置
        print(f"Gene expression level for each downstream gene")
        gene_list = list(self.__downstream_genelist[gene])
        num_genes = len(gene_list)
        cols = 6
        rows = (num_genes + cols - 1) // cols  # Round up to the nearest integer
        plt.figure(dpi=300)
        gs = plt.GridSpec(rows, cols)  # Create a grid for subplots
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams.update({'font.size': 3})
        for i, gene_name in enumerate(gene_list):
            ax = plt.subplot(gs[i])  # Create a subplot in the grid
            data = self.__gene_exp[gene_name]
            sns.distplot(data, axlabel=gene_name, ax=ax)
            ax.axvline(self.__gene_exp.loc[gene, gene_name], color='red', lw=0.5)
        # Remove any empty subplots in the grid
        for i in range(num_genes, rows * cols):
            plt.delaxes(plt.subplot(gs[i]))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.tick_params(pad=0.1)
        plt.show()
    


    def generate_global_stats(self):
        
        # ### sequencing quality
        # print(f"Overview of sequencing quality:")
        # zero_per=self.__expression.eq(0).sum()/self.__expression.shape[0]
        # plt.rcParams.update({'font.size':10})
        # sns.displot(zero_per)
        # plt.xlabel("zero percent of each cell")
        # print(f"mean zero_percent for all cell:",zero_per.mean())
        # plt.show()
        
        ## cell number distribution
        print(f"Distribution of cell number:")
        sns.distplot(np.log2(self.__perturb_fliter['cell_number']))
        plt.axvline(x=np.log2(self.__perturb_fliter.loc['non-targeting','cell_number']), color='red', label='non-targeting')
        plt.xlabel('log2(cell_number)')
        plt.title('Perturtion cell number distribution')
        plt.legend()
        plt.show()
        print()

        ### plotg scatter: cell_number--total_mrna_cv  (the basis of filter accroding to cell_number)
        print(f"Relationship between cell number and total mRNA CV:")
        sns.scatterplot(np.log2(self.__perturb_fliter['cell_number']),self.__perturb_fliter['total_rna_cv'])
        plt.axvline(x=4, color='red', label='threshold')
        plt.xlabel('log2(cell_number)')
        plt.legend()
        plt.show()
        print()

        ### total mRNA distribution visualization
        print(f"Total mRNA distribution:")
        sns.distplot(self.__total_mRNA['total_mRNA'])
        plt.axvline(self.__total_mRNA.loc['non-targeting','total_mRNA'], color='red', label='non-targeting')
        plt.legend()
        plt.show()

        
        ### info of knocked gene
        print(f"Details of the knocked-down genes:")
        print(f"Information on genes with higher expression after knock-down:")
        abnormal=self.__knock_gene_df[self.__knock_gene_df['variation']>0]
        print(abnormal)

        print(f"Perturbation with zero expression after knockdown of perturbed genes:")
        to_zero=self.__knock_gene_df[self.__knock_gene_df['knocked']==0]
        sns.displot(to_zero['original'])
        plt.title('Perturbation zero expression after knockdown')
        plt.xlabel('original gene counts')
        plt.show()
        print(f'gene number(after perturbation expression=0):{len(to_zero.index)}')
        print('gene list(after perturbation expression=0):')
        print(to_zero.index)


def global_scatter(analyse):
    # import matplotlib
    # import plotly.graph_objects as go
    # %matplotlib
    ### global scatter plot


    total_mRNA=analyse.total_mRNA_info
    knock_gene_df=analyse.knocked_gene_info
    total_mRNA_hue=total_mRNA[total_mRNA.index.isin(knock_gene_df.index)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6), gridspec_kw={'width_ratios': [3, 2]})
    ax1.tick_params(axis='both',)
    ax1.set_title('original VS knocked-down gene expression level comparison')
    plt.rcParams.update({'font.size': 10})
    # scatter plot
    sc = ax1.scatter(np.log2(knock_gene_df['original']),np.log2(knock_gene_df['knocked']),c=total_mRNA_hue['ratio'].rename('total mRNA ratio'),cmap='bwr')
    ax1.plot([-9,max(np.log2(knock_gene_df['original']))],[-9,max(np.log2(knock_gene_df['original']))],color='gray',linestyle='--')
    # cax=plt.axes([0.95,0.1,0.03,0,8])
    cbar=plt.colorbar(sc)
    # cbar.set_lable('Color value')
    # axis 1 labels
    ax1.set_xlabel('log2(original gene expression)')
    ax1.set_ylabel('log2(knocked-down gene experssion)')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Detail info')
    ax2.set_xlim(0,2)

    cursor = mplcursors.cursor(sc, hover=True)# by default the annotation displays the xy positions
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(knock_gene_df.index[sel.index])
        for text in ax2.texts:
            text.set_visible(False)
        source =knock_gene_df.index[sel.index]
        # ax2.text(0.03, 0.2, source, ha='left', va='top', fontsize=7)
        ax2.axhline(0.3,lw=0.2,color='black',linestyle='--')
        ax2.text(0.03,0.9,"original expresison:")
        ax2.text(2,0.9,analyse._SCPerturbationAnalysis__gene_exp.loc['non-targeting',knock_gene_df.index[sel.index]],ha='right')
        ax2.text(0.03,0.8,"knocked expresison:")
        ax2.text(2,0.8,analyse._SCPerturbationAnalysis__gene_exp.loc[knock_gene_df.index[sel.index],knock_gene_df.index[sel.index]],ha='right')
        ax2.text(0.03,0.7,"original total mRNA:")
        ax2.text(2, 0.67, total_mRNA.loc['non-targeting','total_mRNA'],ha='right')
        ax2.text(0.03,0.6,"knocked total mRNA:")
        ax2.text(2,0.57, total_mRNA.loc[knock_gene_df.index[sel.index],'total_mRNA'],ha='right')  
        ax2.text(0.03,0.5,"amount of downstream genes:")
        ax2.text(2,0.5,analyse._SCPerturbationAnalysis__downstream_counts.loc[knock_gene_df.index[sel.index],'counts'],ha='right')
        ax2.text(0.03,0.4,"amount of upstream genes:")
        ax2.text(2,0.4,analyse.uptream_gene_counts.iloc[sel.index,0],ha='right')
        ax2.text(1,0.25,"Colors in scatterplot: total mRNA ratio",ha='center')
        ax2.text(1,0.15,"For more detailed infomation,",ha='center')
        ax2.text(1,0.12,"please run:",ha='center')
        ax2.text(1,0.09,"_.get_all_info_for_perturbation(gene)",ha='center')
    plt.legend()
    plt.show()

def scatter_vocano_for_gene(analyse, gene, color_gene_list, output_expression_is_0):
    ### scatter and vocano plot for each perturbation
    total_mRNA=analyse.total_mRNA_info
    expression=analyse._SCPerturbationAnalysis__expression
    anndata=analyse._SCPerturbationAnalysis__anndata
    ##### downstream_gene_list=analyse.downstream_gene_list
    gene_exp=analyse.gene_exp


    expriment=expression[anndata.obs[anndata.obs['gene']==gene].index]
    control=expression[anndata.obs[anndata.obs['gene']=='non-targeting'].index].mean(axis=1)
    control.index=analyse.gene_lfc.columns
    # %matplotlib inline
    # plt.hist(control)
    # plt.show()


    # %matplotlib
    # control=self.__self.__expression[self.__self.__anndata.obs[self.__self.__anndata.obs['gene']=='non-targeting'].index].sample(n=expriment.shape[1],axis=1).mean(axis=1)
    expriment1=expriment.mean(axis=1)
    expriment1.index=analyse.gene_lfc.columns
    print(f"knocked gene", gene)
    print(f"cell number:",expriment.shape[1])

    ### scatter plot and valcano
    fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    plt.rcParams.update({'font.size': 10})
    sct=ax1.scatter(np.log2(control),(np.log2(expriment1)),s=10)
    ###### downstream_genes=downstream_gene_list[gene]
    ax1.scatter(np.log2(control[control.index.isin(color_gene_list)]),(np.log2(expriment1[control.index.isin(color_gene_list)])),edgecolors='orange',s=10, label='downstream genes')
    ax1.scatter(np.log2(control[gene]),(np.log2(expriment1[gene])),color='red',s=10, label='knocked-down gene')
    cursor = mplcursors.cursor(sct, hover=True)
    # by default the annotation displays the xy positions
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text([control.index[sel.index],gene_exp.loc[gene,control.index[sel.index]]
                                ,gene_exp.loc['non-targeting',control.index[sel.index]]])

    ax1.plot([np.log2(min(control)),np.log2(max(control))],[np.log2(min(control)),np.log2(max(control))],color='red',linestyle='--',label='y=x')
    ratio=total_mRNA.loc[gene,'ratio']
    ax1.plot([np.log2(min(control)),np.log2(max(control))],[np.log2(min(control))*ratio,np.log2(max(control))*ratio],color='red',label=f"total_mRNS_ratio={ratio:.2f}")
    ax1.set_xlabel('log2(original UMI count)')
    ax1.set_ylabel('log2(perturbed UMI count)')
    ax1.set_title('original VS knocked-down gene expression')
    ax1.legend(loc='upper left',bbox_to_anchor=(0,1), frameon=False)
    # return 敲完表达为0的基因，敲完表达变化大的基因


    p_values=analyse.gene_pvalue.loc[gene]
    log2_fold_change = analyse.gene_lfc.loc[gene]
    data = pd.DataFrame({'log2_fold_change': log2_fold_change, 'p_value': p_values},index=analyse.gene_lfc.columns)
    p_value_threshold = 0.05
    fold_change_threshold = 1.5
    voc=ax2.scatter(data['log2_fold_change'], -np.log10(data['p_value']),color='gray', alpha=0.7)
    plt.scatter(data[data.index.isin(color_gene_list)]['log2_fold_change'], -np.log10(data[data.index.isin(color_gene_list)]['p_value']), color='red', alpha=0.7)
    cursor = mplcursors.cursor(voc, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(data.index[sel.index])
    ax2.axhline(-np.log10(p_value_threshold), color='blue', linestyle='--', label='p-value threshold')
    ax2.axvline(fold_change_threshold, color='green', linestyle='--', label='fold change threshold')
    ax2.axvline(-fold_change_threshold, color='green', linestyle='--')
    ax2.set_xlabel('log2 Fold Change')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Gene Volcano Plot')
    if output_expression_is_0==True:
        expr=pd.DataFrame(expriment1,columns=['UMI'])
        zero_gene=expr[expr['UMI']==0].index
        print(f'genes with 0 expression after perturbation)')
        return zero_gene
