import numpy as np
import matplotlib.pyplot as plt


dir_result = []

dir_noise_obj_dict = {}
dir_noise_fidel_dict = {}

dm_result = []

dm_noise_obj_dict = {}
dm_noise_fidel_dict = {}


noise_list = [10**(-6), 10**(-5), 10**(-4),
                  5*10**(-4), 10**(-3), 2*10**(-3),
                  5*10**(-3), 10**(-2)]


noise_list_str = [str(noise) for noise in noise_list]


with open("h4_varnoise_dir_depolarize.txt") as file:
    dir_temp_str = [line.rstrip() for line in file]
    dir_temp = []
    for line_str in dir_temp_str:
        line_parse = line_str.split(' ')[-3:]
        line_data = [float(str_data) for str_data in line_parse]
        dir_temp.append(line_data)
    dir_result = dir_result + dir_temp




with open("h4_varnoise_dm_depolarize.txt") as file:
    dm_temp_str = [line.rstrip() for line in file]
    dm_temp = []
    for line_str in dm_temp_str:
        line_parse = line_str.split(' ')[-3:]
        line_data = [float(str_data) for str_data in line_parse]
        dm_temp.append(line_data)
    dm_result = dm_result + dm_temp








for noise, noise_str in zip(noise_list, noise_list_str):
    dir_obj_temp = []
    dir_fidel_temp = []
    for result_line in dir_result:
        if np.abs(noise-result_line[0])<1e-9:
            dir_obj_temp.append(result_line[-2])
            dir_fidel_temp.append(result_line[-1])
    dir_noise_obj_dict[noise_str] = dir_obj_temp
    dir_noise_fidel_dict[noise_str] = dir_fidel_temp

for noise, noise_str in zip(noise_list, noise_list_str):
    dm_obj_temp = []
    dm_fidel_temp = []
    for result_line in dm_result:
        if np.abs(noise-result_line[0])<1e-9:
            dm_obj_temp.append(result_line[-2])
            dm_fidel_temp.append(result_line[-1])
    dm_noise_obj_dict[noise_str] = dm_obj_temp
    dm_noise_fidel_dict[noise_str] = dm_fidel_temp


dir_obj_mean_list = []
dir_obj_std_list = []
dir_obj_full = []
dir_fidel_mean_list = []
dir_fidel_std_list = []
dir_fidel_full = []
dm_obj_mean_list = []
dm_obj_std_list = []
dm_obj_full = []
dm_fidel_mean_list = []
dm_fidel_std_list = []
dm_fidel_full = []

for noise, noise_str in zip(noise_list, noise_list_str):
    dir_obj_temp_arr = np.array(dir_noise_obj_dict[noise_str])
    dir_obj_full.append(dir_obj_temp_arr)
    dir_obj_mean_list.append(np.mean(dir_obj_temp_arr))
    dir_obj_std_list.append(np.std(dir_obj_temp_arr))
    dir_fidel_temp_arr = np.array(dir_noise_fidel_dict[noise_str])
    dir_fidel_full.append(dir_fidel_temp_arr)
    dir_fidel_mean_list.append(np.mean(dir_fidel_temp_arr))
    dir_fidel_std_list.append(np.std(dir_fidel_temp_arr))
    dm_obj_temp_arr = np.array(dm_noise_obj_dict[noise_str])
    dm_obj_full.append(dm_obj_temp_arr)
    dm_obj_mean_list.append(np.mean(dm_obj_temp_arr))
    dm_obj_std_list.append(np.std(dm_obj_temp_arr))
    dm_fidel_temp_arr = np.array(dm_noise_fidel_dict[noise_str])
    dm_fidel_full.append(dm_fidel_temp_arr)
    dm_fidel_mean_list.append(np.mean(dm_fidel_temp_arr))
    dm_fidel_std_list.append(np.std(dm_fidel_temp_arr))




color_dict = {'Blue':'#3366CC',
              'SkyBlue':'#0099C6',
              'Teal':'#22AA99',
              'Red':'#DC3912',
              'FireBrick':'#B82E2E',
              'Pink':'#DD4477',
              'Orange':'#FF9900',
              'DeepOrange':'#E67300',
              'Green':'#109618',
              'LightGreen':'#66AA00',
              'Purple':'#990099'}


plt.figure()
plt.plot(noise_list, np.median(dir_obj_full, axis=1),
          color = color_dict['Blue'],
          label = 'Traditional VQE',
         marker = 'o', markersize = 4)
plt.fill_between(noise_list, np.min(dir_obj_full, axis=1),
                             np.max(dir_obj_full, axis=1),
                              color = color_dict['Blue'],
                              alpha = 0.4)
plt.plot(noise_list, np.median(dm_obj_full, axis=1),
          color = color_dict['DeepOrange'],
          label = 'Our method',
         marker = '*', markersize = 4)
plt.fill_between(noise_list, np.min(dm_obj_full, axis=1),
                             np.max(dm_obj_full,axis=1),
                              color = color_dict['DeepOrange'],
                              alpha = 0.5)

plt.grid(True)
plt.xlabel('Noise strength')
plt.ylabel('Ground state energy')
plt.xscale('logit')
plt.legend()
plt.savefig('Molecule_obj.pdf',bbox_inches = "tight")


fig, ax1 = plt.subplots()

ax1.plot(noise_list, np.median(dir_fidel_full, axis=1),
          color = color_dict['Blue'],
          label = 'Traditional VQE',
         marker = 'o', markersize = 4)
ax1.fill_between(noise_list, np.min(dir_fidel_full, axis=1),
                             np.max(dir_fidel_full, axis=1),
                              color = color_dict['Blue'],
                              alpha = 0.4)
ax1.plot(noise_list, np.median(dm_fidel_full, axis=1),
        color = color_dict['DeepOrange'],
        label = 'Our method',
         marker = '*', markersize = 4)


ax1.fill_between(noise_list, np.min(dm_fidel_full, axis=1),
                             np.max(dm_fidel_full, axis=1),
                              color = color_dict['DeepOrange'],
                              alpha = 0.5)
plt.grid(True)
plt.xlabel('Noise strength',fontsize=14)
plt.ylabel('Ground state fidelity',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale('logit')
plt.yscale('logit')
plt.legend()
plt.savefig('Molecule_fidelity.pdf',bbox_inches = "tight")



noise_dict={}
noise_list_pure = [[] for i in noise_list_str]

with open('h4_varnoise_dm_depolarize_log.txt') as file:
    line_list = [line.rstrip() for line in file]
noise_index_list = []

for index, line in zip(range(len(line_list)), line_list):
    if len(line) < 70:
        noise_index_list.append(index)
for pure_list, noise_str in zip(noise_list_pure, 
                                      noise_list_str):
    for noise_index, place_index in zip(noise_index_list,
                             range(len(noise_index_list))):
        line_temp = line_list[noise_index]
        noise_str_temp = line_temp.split(' ')[-1]
        if noise_str_temp == noise_str:
            if place_index == len(noise_index_list)-1:
                if noise_index_list[place_index] != len(line_list):
                    for row_index in range(
                        noise_index_list[place_index]+1, 
                                           len(line_list)):
                        line_append = line_list[row_index]
                        line_split = line_append.split(' ')[-4:]
                        for item in line_split:
                            pure_list.append(float(item))
            else:
                for row_index in range(noise_index_list[place_index]+1, 
                                    noise_index_list[place_index+1]):
                    
                    line_append = line_list[row_index]
                    line_split = line_append.split(' ')[-4:]
                    for item in line_split:
                        pure_list.append(float(item))




        
actuall_pure_list = noise_list_pure


median_pure_list = [np.median(np.array(list_temp)) 
                    for list_temp in actuall_pure_list]

position_list = noise_list



fig, ax = plt.subplots()
position_list = np.array(position_list)
color = 'lightblue'
color = '#5FBCD3'
color_line = '#FF7F2A'
marker_color = '#BCD35F'
flierprops = dict(marker='o', markeredgecolor=marker_color)
w = 0.2
width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
medianprops = dict(linestyle='-', linewidth=2.5, color=color_line)
boxprops = dict(linestyle='-', linewidth=1.8, color=color)
bplot = ax.boxplot(actuall_pure_list,
            vert=True,
            patch_artist=True,
            positions=position_list,
            medianprops=medianprops,
            flierprops=flierprops,
            whis=(1,99),
            boxprops=boxprops,
            widths=width(position_list,w))
ax.plot(position_list, median_pure_list,
        marker='.', linestyle = 'dashed', color=color_line,
        linewidth=1)
plt.grid(True)
plt.xlabel('Noise strength', fontsize=14)
plt.ylabel('Purity', fontsize=14)
plt.xscale('logit')
plt.yscale('logit')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ticks_y = ax.yaxis.get_minor_ticks()
labels = [tick.get_loc() for tick in ticks_y]
labels_change = [label if (label > 0.1 and label < 0.4) or \
                 (label > 0.6 and label < 0.9) \
                    else '' for label in labels]

plt.yticks(ticks=labels,
           labels=labels_change, minor=True,
            fontsize=12)


plt.xlim(right=0.015)
plt.setp(bplot['whiskers'], color=color, lw=2)
plt.setp(bplot['caps'], color=color, lw=2)

for patch in bplot['boxes']:
    patch.set_facecolor(color)
plt.savefig('Molecule_purity.pdf',bbox_inches = "tight")

plt.show()
