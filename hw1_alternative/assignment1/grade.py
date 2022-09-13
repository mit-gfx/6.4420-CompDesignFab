import os
import argparse
import numpy as np

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Grade HW1')
    parser.add_argument('--section', metavar='SECTION_NAME', default='all',
                        choices=['brute_force', 'accelerated', 'non_watertight', 'all'],
                        help='Optionally specify which section to grade')

    args = parser.parse_args()

    # Print welcome message
    print('Start HW1 grading')

    # Section information table
    section_menu = {
        'brute_force': {
            'name': 'Brute force voxelization',
            'cases': ['spot', 'bunny', 'fandisk', 'dragon']
        },
        'accelerated': {
            'name': 'Accelerated voxelization',
            'cases': ['spot_fast', 'bunny_fast', 'fandisk_fast', 'dragon_fast']
        },
        'non_watertight': {
            'name': 'Voxelization of non-watertight meshes',
            'cases': ['spot_with_hole', 'bunny_with_hole']
        }
    }

    # Locate result folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(root_dir, 'data', 'assignment1')

    # Determine test sections
    test_sections = section_menu.keys() if args.section == 'all' else [args.section]

    for section in test_sections:
        print('=' * 20)
        print(section_menu[section]['name'])

        # Get folder names and mesh names
        folder_names = section_menu[section]['cases']
        cases = folder_names
        if section == 'accelerated':
            cases = [name[:name.find('_')] for name in cases]

        # Loop over each case
        for folder_name, mesh_name in zip(folder_names, cases):
            try:
                # Read student result
                data_stu = np.load(os.path.join(result_dir, 'results', folder_name, f'{mesh_name}_voxel_data.npz'))
                voxels_stu = data_stu['voxels']
            except FileNotFoundError:
                print(f'  {folder_name:<20s}:  Student result not found')
                continue

            try:
                # Read reference result
                data_ref = np.load(os.path.join(result_dir, 'references', mesh_name, f'{mesh_name}_voxel_data.npz'))
                voxels_ref = data_ref['voxels']
            except FileNotFoundError:
                print(f'  {folder_name:<20s}:  Reference result not found')
                continue

            # Check if they are of equal shapes
            if voxels_ref.shape != voxels_stu.shape:
                print(f'  {folder_name:<20s}:  Incorrect voxel grid dimensions')
                continue

            # Compute IoU
            iou = np.sum(voxels_ref & voxels_stu) / np.sum(voxels_ref | voxels_stu)

            # Compute difference
            diff = np.sum(voxels_ref ^ voxels_stu)
            suffix = f'  ({diff} voxels different)' if diff else ''

            print(f'  {folder_name:<20s}:  similarity = {iou * 100:.6g} %{suffix}')


if __name__ == '__main__':
    main()
