import numpy as np
import GPy
import json
from matplotlib import pyplot as plt
from sklearn import preprocessing
import os
plt.ioff()

def plot_kernel(imageFileName, s, xmax, reverse=True):
	#plot figure
	plt.rcParams['figure.dpi'] = 300
	fig = plt.figure(figsize=(20,20))
	#fig = plt.figure()
	vmin = np.amin(s)
	vmax = np.amax(s)

	if(reverse):
		s = np.flip(s.T, 0)	# (0, 0) on left-down
	
	plt.matshow(s.reshape((xmax, xmax)), interpolation='none', cmap=plt.get_cmap('Spectral_r'), vmin=vmin, vmax=vmax)
	plt.colorbar(label='m(x)', shrink=0.75)
	plt.xticks(np.arange(0.5,xmax+0.5,1), [])
	plt.yticks(np.arange(0.5, xmax+.5,1), [])
	plt.tick_params(axis='both', which='both',length=0)
	plt.grid(color='#0082C1', linestyle='-', linewidth=0.5)
	s_reshape = np.round(s.reshape((xmax, xmax)), 2)
	font_size = {5: 10, 8:8, 10: 7, 20: 5, 30: 2, 40: 2, 50: 1}
	for i in range(xmax):
		for j in range(xmax):
			plt.text(i, xmax-1-j, s_reshape[xmax-1-j, i], ha='center', fontsize=font_size[xmax])

	#save fig
	plt.savefig(imageFileName, bbox_inches='tight', format='png')
	plt.close(fig)
	plt.close()

new_method = False
if __name__ == '__main__':
    g = 40
    land_l = 3.0
    n = 10
    if new_method:
        file = os.path.join('landscape', f'Grid{g}_Lambda{land_l}_newmethod',f'Grid{g}_Lambda{land_l}_{n}.json')
    else:
        file = os.path.join('landscape', f'Grid{g}_Lambda{land_l}', f'Grid{g}_Lambda{land_l}_{n}.json')

    with open(file, "r") as f:
        json_f = json.load(f)
        landscape = np.reshape(np.asarray(json_f["Landscape"]), (g, g))

    # print(np.round(landscape, 2))
    output_folder = os.path.join('landscape', 'image_output')
    os.makedirs(output_folder, exist_ok=True)
    if new_method:
          output_file = os.path.join(output_folder, f'Grid{g}_Lambda{land_l}_{n}_newmethod.png')
    else:
           output_file = os.path.join(output_folder, f'Grid{g}_Lambda{land_l}_{n}.png')
    plot_kernel(output_file, landscape, g, reverse=True)