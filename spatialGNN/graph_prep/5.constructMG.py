# Construct meta graph
import argparse
import pandas as pd
import networkx as nx
import obonet
import scanpy as sc

from utils import load_celltype_ppi


OUTPUT_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/networks"
BTO_DIR = "/n/data1/hms/dbmi/zitnik/lab/datasets/2022-10-BTO/data/BTO.obo"
# BTO_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/ontologies/uberon.obo"


def filter_cci(cci_f, celltype_ppi):
    cci = nx.read_edgelist(cci_f, delimiter = "\t", create_using = nx.MultiGraph)
    print(nx.info(cci))

    celltypes = [c[1].lower() for c in celltype_ppi]
    cci = cci.subgraph(celltypes)
    print("Filtered cell-cell interactions...")
    print(nx.info(cci))

    assert len(cci.nodes) == len(celltypes)
    return cci


def read_tissue_metadata(f, annotation, celltypes):
    if type(f) == pd.core.frame.DataFrame:
        tissue_metadata = f
    else:
        tissue_metadata = pd.read_csv(f, sep ="\t")

    if "orig_tissue" in tissue_metadata.columns:
        tissue_metadata["organ_tissue"] = tissue_metadata["orig_tissue"] 

    celltype2tissue = dict()
    for c in tissue_metadata[annotation].unique():
        if c not in celltypes: continue
        c_tissue = tissue_metadata[tissue_metadata[annotation] == c]["organ_tissue"].unique()
        if c in celltype2tissue: 
            assert celltype2tissue[c] == c_tissue.tolist(), c
        celltype2tissue[c] = c_tissue.tolist()

    return celltype2tissue, tissue_metadata["organ_tissue"].unique()


def extract_BTO(unique_tissues):
    bto_G = obonet.read_obo(BTO_DIR)
    print(nx.info(bto_G))

    name2bto = {data.get('name'): id_ for id_, data in bto_G.nodes(data=True)} 
    manual_mapping = {"mammary": "mammary gland", 
                      "fat": "adipose tissue", 
                      "prostate": "prostate gland",
                      'esophagus muscularis mucosa': 'muscularis mucosa',
                      'lens of camera-type eye': 'lens',
                      'transverse colon': 'colon transversum',
                      'lingula of left lung': 'lung',
                      'body of stomach': 'gastric corpus',
                      'heart right ventricle': 'right ventricle',
                      'cardiac atrium': 'atrium',
                      'omentum': 'peritoneum',
                      'rectus abdominis muscle': 'abdominal muscle',
                      'apex of heart': 'heart',
                      'caudate lobe of liver': 'liver',
                      'bladder organ': 'bladder',
                      'eye trabecular meshwork': 'reticulum trabeculare',
                      'cerebral nuclei': 'cerebral nucleus',
                      'Brodmann (1909) area 4': "motor cortex",
                      'brodmann (1909) area 4': "motor cortex",
                      'endometrium': 'uterine endometrium',
                      'heart left ventricle': 'right ventricle',
                      'sigmoid colon': 'colon sigmoideum',
                      'fovea centralis': 'retina',
                      'muscle organ': 'muscle',
                      'white matter of cerebellum': 'cerebellum',
                      'ampulla of uterine tube': 'oviductal ampulla',
                      'thoracic lymph node': 'lymph node',
                      'skin of leg': 'skin',
                      'skeletal muscle tissue': 'skeletal muscle',
                      'muscle of pelvic diaphragm': 'diaphragm',
                      'skin of abdomen': 'skin',
                      'cervical spinal cord white matter': 'spinal cord',
                      'epithelium of trachea': 'tracheal epithelium',
                      'corneo-scleral junction': 'sclera',
                      'macula lutea proper': 'macula lutea',
                      'basal zone of heart': 'heart',
                      'inguinal lymph node': 'lymph node',
                      'caecum': 'cecum',
                      'respiratory airway': 'respiratory epithelium',
                      'peripheral region of retina': 'retina',
                      'muscle of abdomen': 'abdominal muscle',
                      'retinal neural layer': 'neural retina',
                      'cardiac ventricle': 'heart ventricle',
                      'skin of chest': 'skin',
                      'pyloric antrum': 'gastric antrum',
                      'transition zone of prostate': 'prostate gland',
                      'isthmus of fallopian tube': 'oviduct',
                      'subcutaneous abdominal adipose tissue': 'abdominal adipose tissue',
                      'fimbria of uterine tube': 'oviduct',
                      'pigment epithelium of eye': 'retinal pigment epithelium',
                      'visceral abdominal adipose tissue': 'abdominal adipose tissue',
                      'cortex of kidney': 'renal cortex',
                      'muscle tissue': 'muscle',
                      'skin of body': 'skin',
                      'fallopian tube': 'oviduct',
                      'anterior part of tongue': 'tongue',
                      'hippocampal formation': 'hippocampus',
                      'thalamic complex': 'thalamus',
                      'lung parenchyma': 'alveolar epithelium',
                      'peripheral zone of prostate': 'prostate gland',
                      'posterior part of tongue': 'tongue',
                      'anterior wall of left ventricle': 'left ventricle'}

    bto_subgraph = set() # Nodes in the BTO to extract (relevant to our cell type PPI networks)
    tissue2bto = dict() # Mapping from our tissue names to node IDs in BTO

    for t in unique_tissues:
        orig_t = t
        t = " ".join(t.split("_")).lower()
        if t not in name2bto: t = manual_mapping[t]

        t_descendants = all_descendants(bto_G, name2bto[t])
        #print(t, t_descendants)

        bto_subgraph = bto_subgraph.union(t_descendants)
        tissue2bto[orig_t] = name2bto[t]
    
    bto_subgraph = bto_G.subgraph(bto_subgraph) 
    
    print("Number of tissues", len(unique_tissues))
    print("Number of BTO nodes", len(bto_subgraph.nodes)) 
    print("Number of BTO edges", len(bto_subgraph.edges)) 
    return bto_subgraph, tissue2bto


def all_descendants(G, term):
    paths = nx.all_simple_paths(G, term, "BTO:0000000") # BTO:0000000 is the "tissues, cell types and enzyme sources" term 
    all_d = set()
    for p in paths:
        all_d = all_d.union(set(p)) 
    return all_d


def create_ct_graph(celltype2tissue, tissue2bto):
    ct_edgelist = []
    for c, c_tissue in celltype2tissue.items():
        for t in c_tissue:
            ct_edgelist.append((c, tissue2bto[t]))
    
    print("Number of cell-tissue edges:", len(ct_edgelist))
    return ct_edgelist


def main():

    parser = argparse.ArgumentParser(description="Constructing meta graph.")
    parser.add_argument("-counts", type=str, help="Ranked h5ad.")
    parser.add_argument("-celltype_ppi", type=str, help="Filename (prefix) of cell type PPI.")
    parser.add_argument("-annotation", type=str, default="cell_ontology_class", help="Column for cell type annotation.")
    parser.add_argument("-cci_edgelist", type=str, help="Filename of cell-cell interaction network.")
    parser.add_argument("-mg_edgelist", type=str, help="Filename of meta graph.")
    args = parser.parse_args()

    # Read cell type PPI networks
    celltype_ppi = load_celltype_ppi(args.celltype_ppi)
    print("Number of cell types with PPI networks:", len(celltype_ppi))

    # Filter cell-cell interaction network to only include cell types with PPI networks
    cci = filter_cci(args.cci_edgelist, celltype_ppi)

    # Read tissue meta data for cells
    if args.counts:
        adata = sc.read_h5ad(args.counts, backed="r")
        celltype2tissue, unique_tissues = read_tissue_metadata(adata.obs, args.annotation, cci.nodes())
    else:
        celltype2tissue, unique_tissues = read_tissue_metadata(OUTPUT_DIR + "ts_data_tissue.csv", args.annotation, cci.nodes())
    print("Number of unique tissues:", len(unique_tissues))
    
    # Extract relevant tissues from BTO
    bto_G, tissue2bto = extract_BTO(unique_tissues)

    # Create cell-tissue graph
    ct_edgelist = create_ct_graph(celltype2tissue, tissue2bto)

    # Combine cell-cell, cell-tissue, tissue-tissue graphs
    metagraph = nx.union_all([cci, bto_G])
    print("CCI + BTO:\n", nx.info(metagraph))
    metagraph.add_edges_from(ct_edgelist)
    print("Meta graph:\n", nx.info(metagraph))
    
    new_metagraph = nx.Graph()
    new_metagraph.add_edges_from(list(metagraph.edges()))
    print("New meta graph (checking that it's the same as the meta graph):\n", nx.info(new_metagraph))
    
    # Save
    nx.write_edgelist(metagraph, args.mg_edgelist, data = False, delimiter = "\t")


if __name__ == "__main__":
    main() 
