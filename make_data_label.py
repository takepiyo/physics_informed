import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Num_stencil = 100
Initial_boundary = 5
Final_boundary = 25

x_array = np.arange(Num_stencil)
u_array = np.where(((x_array >= Final_boundary) | (x_array < Initial_boundary)), 0.0, 1.0)

#plt.plot(u_array)

Delta_x = 0.1
Delta_t = 0.1
c = 0.5
CFL = c * Delta_t / Delta_x
print(CFL)

ims = []
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-0.25, 1.25)

exact_list = []
time_list = []

Time_step = 100
for t in range(Time_step):
    time_list.append(Delta_t * t)
    total_movement = c * (Delta_t)/(Delta_x) * t  #x_index_increased
    #print('total_mocement: ', total_movement)
    exact_u_array = np.where(((x_array >= Final_boundary + total_movement) | (x_array < Initial_boundary + total_movement)), 0.0, 1.0)
    #print(exact_u_array.shape)
    exact_list.append(exact_u_array[np.newaxis, :])
    #im = ax.plot(x_array * 0.1, exact_u_array, color='b')
    #if t == 50:
        #im = ax.plot(x_array * 0.1, exact_u_array, color='b')
        #plt.show()
    #print(type(im))
    #ims.append(im)
    #break

data_label = np.stack(exact_list, axis=0).transpose(2, 1, 0)

time_array = np.array(time_list).reshape(-1, 1)
x_pos_array = np.arange(start=0.0, stop=10.0, step=0.1).reshape(-1, 1)

print(x_pos_array.shape)
print('='*80)

print(time_array.shape)
print('-'*80)
print(data_label[50])
print(data_label.shape)
np.save('data_label.npy', data_label)

ani = animation.ArtistAnimation(fig, ims, interval = 100)
#plt.show()

dataset_dict = {'t': time_array, 'x': x_pos_array, 'u': data_label}
plt.imshow(data_label[:][0][:].transpose())
plt.show()

#import pickle
#with open('dataset.pkl', mode='wb') as f:    
    #pickle.dump(dataset_dict, f)


