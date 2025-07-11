from read_file_meta_utils import *
from file_utils import *
# from s3file_utils import *
from asin_utils import (
    get_silver_span,
    get_packsize,
    standardize_brand,
    predict_sub_brand_ml,
    predict_size_with_crf
)
import pandas as pd  
import logging
import os
import re

def keep_numbers(text):
    """Keeps only numbers in a string."""
    return ''.join(filter(str.isdigit, text))


def read_asin(metadata_file_path=None, metadata_file_name=None, sheetname=None, **kwargs):
    logging.info(f"entered read_asin")
    
    # Check if we should use local CSV (for testing/development)
    use_local_csv = kwargs.get('use_local_csv', True)  # Default to local for now
    
    if use_local_csv:
        print("Loading data from local CSV file...")
        asin_pdf = pd.read_csv(r"C:\Users\AmosXiao\Desktop\all_asins.csv", dtype=str).fillna("")
        
        # Rename columns to match expected format
        asin_pdf = asin_pdf.rename(columns={
            'Brand Name': 'brand_name',
            'Item Name': 'item_name', 
            'Subcategory Label': 'subcategory_label',
            'Category Label': 'category_label',
            'Category Code': 'category_code'
        })
        
    else:
        # S3 MODE: Original metadata-driven approach
        if metadata_file_name and metadata_file_path:
            logging.debug(f"{locals()}")
            
            file_loc, colmap = read_sourcefile_loc_colmap(
                metadata_file_path, metadata_file_name,
                sheetname='fileloc', sourcetype='asin_mapping'
            )
            meta_pd = read_metadata_file(
                metadata_file_path, metadata_file_name,
                sheetname=colmap
            )
            (dtype_dict, rename_dict, prod_agg_dict, geog_agg_dict) = \
                build_input_string_rename_columns(meta_pd, get_agg_rules=True)
            
            dict_for_file = {
                'category': kwargs.get('category', None),
                'filetype': kwargs.get('filetype', None),
                'extension': kwargs.get('extension',   None),
                'brand':     kwargs.get('brand',       None),
            }
            logging.debug(dict_for_file)
            file_loc = os.path.join(file_loc, dict_for_file['brand'], 'amazon_asin')
            logging.debug(file_loc)
            dict_for_read_params = {
                'dtype':      dtype_dict,
                'delimiter':  kwargs.get('delimiter', ","),
                'header':     kwargs.get('header',    0),
                'inpsheetname': kwargs.get('inpsheetname', None),
            }
            logging.debug(rename_dict)
            
            # Uncomment when have S3 access:
            # from s3file_utils import concat_csv_s3
            # asin_pdf = concat_csv_s3(
            #     path=file_loc,
            #     expected_columns=None,
            #     dtypedict=dict_for_read_params['dtype'],
            #     column_map=rename_dict,
            #     **kwargs
            # )
            
            # return empty dataframe if S3 mode is requested but not available
            print("S3 mode requested but not available. Use use_local_csv=True")
            return pd.DataFrame()
    
    # ─── Common processing for both modes ────────────────────────────────────
    
    # 1) original_sub_brand
    asin_pdf['original_sub_brand'] = (
        asin_pdf['subcategory_label']
          .fillna('')
          .str.replace(r'^\d+\s*', '', regex=True)
          .replace({'': '--'})
    )
    
    def create_kv_gmc_subcategory(row):
        """Create KV_GMC_SUBCATEGORY based on Category Label or Category Code + Brand"""
        category_label = row.get('category_label', '')
        
        # Remove numbers and clean up category label
        if category_label and category_label.strip() and category_label.strip() != '--':
            # Remove leading numbers 
            cleaned_category = re.sub(r'^\d+\s*', '', category_label).strip()
            if cleaned_category and cleaned_category != '--':
                return cleaned_category
        
        # If category_label is empty or "--", use Category Code + standardized brand
        category_code = row.get('category_code', '')
        brand_name = row.get('brand_name', '')
        
        if category_code and brand_name:
            standardized_brand = standardize_brand(brand_name)
            return f"{category_code} {standardized_brand}"
        
        return ""
    
    asin_pdf['kv_gmc_subcategory'] = asin_pdf.apply(create_kv_gmc_subcategory, axis=1)
    
    # 2) standardize brand
    asin_pdf['brand'] = asin_pdf['brand_name'].apply(standardize_brand)
    
    # 3) final sub_brand via ML/keyword mapping
    asin_pdf['sub_brand'] = asin_pdf.apply(predict_sub_brand_ml, axis=1)
    
    # 4) size via silver-span
    asin_pdf['size'] = asin_pdf['item_name'].apply(lambda x: predict_size_with_crf(x) or "")
    
    # 5) packsize via regex
    asin_pdf['packsize'] = asin_pdf['item_name']\
                              .apply(get_packsize)
    
    # ─── select only the 7 final columns & drop duplicate ASINs ──────────
    out = asin_pdf[[
        'ASIN',
        'brand',
        'original_sub_brand',
        'sub_brand',
        'item_name',
        'size',
        'packsize',
        'kv_gmc_subcategory'
    ]].drop_duplicates(subset=['ASIN'])
    
    out = out.rename(columns={
        'ASIN': 'UPC',
        'brand': 'KV_BRAND',
        'original_sub_brand': 'KV_SUB_BRAND',
        'sub_brand': 'MODELED_SUB_BRAND',
        'item_name': 'ITEM',
        'size': 'BASE_SIZE',
        'packsize': 'PACKSIZE',  
        'kv_gmc_subcategory': 'KV_GMC_SUBCATEGORY'
    })
    
    # Save file
    out.to_csv(r"C:\Users\AmosXiao\Desktop\asin_lookup_ml.csv", index=False)
    
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg = dict(
        category=None,
        filetype=None,
        extension='tsv.zip',
        delimiter='\t',
        brand=None
    )
    df = read_asin(
        metadata_file_path=(
            "C:\\Users\\AmosXiao\\Desktop\\Consulting\\JnJ\\US\\Idap"
        ),
        metadata_file_name="metadata.xlsx",
        sheetname='fileloc',
        **arg
    )
    print(df.head())
