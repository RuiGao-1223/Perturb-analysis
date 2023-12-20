
import scanpy as sc
import pandas as pd
import math
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")


class SCPerturbationAnalyse:
    def __init__(self, anndata):
        self.__anndata = anndata
        self.__perturbation = list(set(anndata.obs['gene']))
        self.__expression = pd.DataFrame(anndata.X, columns=anndata.var.index, index=anndata.obs.index).T
        self.__gene_exp = self._calculate_gene_exp()
        self.__lfc = self._calculate_lfc()
        self.__pvalue = self._calculate_pvalue()
        self.__total_mRNA = self._calculate_total_mRNA()
        self.__knock_gene_df = self.knocked_gene_info
        self.__downstream_counts=pd.DataFrame() 
        self.__downstream_genelist = pd.DataFrame() 

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
    
    def _calculate_pvalue(self):
        non_targeting = self.__expression[self.__anndata.obs[self.__anndata.obs['gene'] == 'non-targeting'].index]
        gene_pvalue = pd.DataFrame()

        for gene in self.__perturbation:
            cells = self.__anndata.obs[self.__anndata.obs['gene'] == gene]
            expression_filtered = self.__expression[cells.index]
            p_values = (stats.ttest_ind(expression_filtered.values.T, non_targeting.values.T)[1] * 8749)
            gene_pvalue[gene] = p_values

        gene_pvalue.index = self.__expression.index
        rename_dict = dict(zip(self.__anndata.var.index, self.__anndata.var['gene_name']))
        rename_dict['ENSG00000284024'] = 'MSANTD7'
        gene_pvalue.rename(index=rename_dict, inplace=True)
        gene_pvalue = gene_pvalue.T
        return gene_pvalue

    def _calculate_total_mRNA(self):
        control = self.__anndata.obs[self.__anndata.obs['gene'] == 'non-targeting']
        total_rna_log = [np.log2(self.__anndata.obs[self.__anndata.obs['gene'] == gene]['UMI_count']).mean() for gene in self.__perturbation]
        p_value = [scipy.stats.ttest_ind(control['UMI_count'], self.__anndata.obs[self.__anndata.obs['gene'] == gene]['UMI_count'])[1] * 2349 for gene in self.__perturbation]
        total_rna = np.exp(total_rna_log)
        total_mRNA = pd.DataFrame(total_rna, index=self.__perturbation, columns=['total_mRNA'])
        total_mRNA['ratio'] = total_mRNA / total_mRNA.loc['non-targeting', 'total_mRNA']
        total_mRNA['logFC'] = np.log2(total_mRNA['ratio'])
        total_mRNA['p_value'] = p_value
        return total_mRNA

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


    def get_class1_TF(self):
        # Calculate the absolute difference between knocked and non-targeting for all pairs (i, j)
        diff = abs(self.__gene_exp.values - self.__gene_exp.loc['non-targeting'].values)
        diff = np.where((self.__gene_exp.values < 1.5) & (self.__lfc.values < -1) | (self.__lfc.values > 1), diff, 0)

        # Find the critical value
        critical = np.percentile(diff[diff > 0], 90)

        # Plot the distribution of log2(abs(knocked-original))
        sns.distplot(np.log2(diff[diff > 0]))
        plt.axvline(np.log2(critical), color='blue')
        plt.xlabel('log2 abs(knocked-original)')
        plt.show()

        # Find the first-class TFs and count their downstream genes
        first_class_TFs = np.sum((diff > critical) & ((self.__lfc.values < -1) | (self.__lfc.values > 1)), axis=1)
        self.__downstream_counts = pd.DataFrame(first_class_TFs, index=self.__lfc.index, columns=['counts'])
        self.__downstream_genelist = pd.DataFrame([self.__lfc.columns[diff[i] > critical].tolist() for i in range(len(self.__lfc))],
                                                index=self.__lfc.index)
        return self.__downstream_counts, self.__downstream_genelist


    def get_all_info(self,gene):
            self.__downstream_counts, self.__downstream_genelist=self.get_class1_TF()
            output_df1=pd.DataFrame(zip(self.__gene_exp.loc[gene].T, self.__gene_exp.loc['non-targeting'].T, self.__lfc.loc[gene].T, self.__pvalue.loc[gene]),columns=['self.__gene_exp','self.__gene_exp_control','self.__lfc','gene_pvalue'], index=self.__lfc.columns).T
            print(f"knocked gene is {gene}, and the basic information is shown as below: ")
            print(output_df1.to_string())
            print()
            print(f"self.__total_mRNA info:")
            print(self.__total_mRNA.loc[gene])
            print()
            print(f"knocked gene info:")
            print(self.__knock_gene_df.loc[gene])
            print()
            print(f"downstream genes counts is: {self.__downstream_counts.loc[gene,'counts']}")
            print(f"downstream genes:")
            print(self.__downstream_genelist.loc[gene][self.__downstream_genelist.loc[gene].notnull()])

            ### 画火山图
            print(f"火山图：")
            plt.rcParams.update({'font.size':10})
            p_values=self.__pvalue.loc[gene]
            log2_fold_change = self.__lfc.loc[gene]
            data = pd.DataFrame({'log2_fold_change': log2_fold_change, 'p_value': p_values})
            p_value_threshold = 0.05
            fold_change_threshold = 1.5
            plt.scatter(data['log2_fold_change'], -np.log10(data['p_value']), c=np.where((data['p_value'] < p_value_threshold) & (np.abs(data['log2_fold_change']) > fold_change_threshold), 'red', 'gray'), alpha=0.7)
            plt.axhline(-np.log10(p_value_threshold), color='blue', linestyle='--', label='p-value threshold')
            plt.axvline(fold_change_threshold, color='green', linestyle='--', label='fold change threshold')
            plt.axvline(-fold_change_threshold, color='green', linestyle='--')
            plt.xlabel('log2 Fold Change')
            plt.ylabel('-log10(p-value)')
            plt.legend()
            plt.title('Volcano Plot')
            plt.show()

            ### 画total mRNA的分布图
            print(f"total mRNA distribution：")
            sns.distplot(self.__total_mRNA['total_mRNA'])
            plt.axvline(self.__total_mRNA.loc[gene,'total_mRNA'], color='red', label=gene)
            plt.show()
            
            ### 画下游基因counts分布图
            print(f"log2(downstream genes count) distribution：")
            sns.distplot(np.log2(self.__downstream_counts['counts']+1))
            plt.axvline(np.log2(self.__downstream_counts.loc[gene,'counts']+1), color='red', label=gene)
            plt.show()

            ### 对于要观测的target gene，观测其下游基因的表达分布及该perturbation在分布中的位置
            print(f"gene expression level for each downstream genes")
            gene_list=list(self.__downstream_genelist.loc[gene][self.__downstream_genelist.loc[gene].notnull()])
            # plt.figure(dpi=600)
            rows=(len(gene_list)//6)+1
            plt.rcParams['xtick.direction']='in'
            plt.rcParams['ytick.direction']='in'
            plt.rcParams.update({'font.size':3})
            plt.margins(x=0.05,y=0.05)
            for i in range(0,len(gene_list)):
                plt.subplot(rows,6,i+1)
                data=self.__gene_exp[gene_list[i]]
                sns.distplot(data,axlabel=gene_list[i])
                plt.axvline(self.__gene_exp.loc[gene,gene_list[i]], color='red', lw=0.5)
            plt.subplots_adjust(hspace=0.5,wspace=0.5)
            plt.tick_params(pad=0.1)
            plt.show()

            return output_df1, self.__downstream_genelist
    