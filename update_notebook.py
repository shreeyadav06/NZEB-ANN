import json

def main():
    try:
        with open('notebooks/NZEB_Climate_Transfer.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        # Update title
        nb['cells'][0]['source'][1] = 'This notebook runs the experiments for the research paper *Transfer Learning for Cross-Climate Building Energy Prediction: Investigating Data Efficiency and Generalization*.\n'
        
        # Update Phase 4
        nb['cells'][9]['source'][0] = '## Phase 4: Transfer Learning (Secondary Scenarios)\n'
        nb['cells'][9]['source'][1] = 'Perform transfer learning for the secondary scenarios using 5-fold CV.\n'
        
        # Update cell 10 source
        for i, line in enumerate(nb['cells'][10]['source']):
            if 'config.TRANSFER_SCENARIOS' in line:
                nb['cells'][10]['source'][i] = line.replace('config.TRANSFER_SCENARIOS', 'config.SECONDARY_SCENARIOS')
            if 'Transfer: ' in line:
                nb['cells'][10]['source'][i] = line.replace('Transfer:', 'Secondary Transfer:')
                
        # Update Phase 5
        nb['cells'][11]['source'][0] = '## Phase 5: Primary Experiment - Data Efficiency\n'
        nb['cells'][11]['source'][1] = 'Evaluate transfer vs scratch on varying fractions of target data for the primary scenario (Tempe -> Cornell).\n'
        
        # Update Phase 6
        nb['cells'][13]['source'][0] = '## Phase 6: Climate Analysis\n'
        nb['cells'][13]['source'][1] = 'Domain justification using climate distance.\n'
        
        with open('notebooks/NZEB_Climate_Transfer.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print("Updated notebook successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
