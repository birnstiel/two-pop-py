import numpy as np
import sys,os,subprocess,shutil
import matplotlib.pyplot as plt 

from matplotlib.widgets import Slider, Button
from string import ascii_letters
from random import choice


def tridag(a,b,c,r,n):
    """
    The tridag routine from Numerical Recipes in F77 rewritten to F95
    rewritten to matlab rewritten to python :-)
    
    where:     a        =    lower diagonal entries
               b        =    diagonal entries
               c        =    upper diagonal entries
               r        =    right hand side vector
               u        =    result vector
               n        =    size of the vectors
    usage:     u = tridag(a,b,c,r,n)
    """
    
    gam = np.zeros(n)
    u   = np.zeros(n)
    
    if b[0]==0.:
        print('tridag: rewrite equations')
        sys.exit(1)
    
    bet = b[0]
    
    u[0]=r[0]/bet
    
    for j in np.arange(1,n):
        gam[j] = c[j-1]/bet
        bet    = b[j]-a[j]*gam[j]
        
        if bet==0:
            print('tridag failed')
            sys.exit(1)
        u[j]   = (r[j]-a[j]*u[j-1])/bet
    
    for j in np.arange(n-2,-1,-1):
        u[j] = u[j]-gam[j+1]*u[j+1]
    return u

def progress_bar(perc,text=''):
    """
    This is a very simple progress bar which displays the given
    percentage of completion on the command line, overwriting its
    previous output.

    Arguments:
    perc    The percentage of completion (float), should be
             between 0 and 100. Only 100.0 finishes with the
             word "Done!".
    text    Possible text for describing the running process.
    
    Example:
    >>> import time
    >>> for i in linspace(0,100,1000):
    >>>     progress_bar(i,text='Waiting')
    >>>     time.sleep(0.005)
    """
    if text!='': text = text+' ... '
    if perc==100.0:
        sys.stdout.write('\r'+text+'Done!\n')
        sys.stdout.flush()
    else:
        sys.stdout.write('\r'+text+'%d %%'%round(perc))
        sys.stdout.flush()
        
class plotter:
    def __init__(self,x,data,y=None,data2=[],data3=[],times=None,i_start=0,xlog=False,ylog=False,zlog=False,xlim=None,ylim=None,zlim=None,xlabel='',ylabel='',lstyle='-',ncont=None,cmap=None,fill=True,ext_link=None):
        """
        creates a GUI to display timedependent 1D or 2D data.
        
        Arguments:
        x    = the x axis array of length nx
        data = - array of the form (nt,nx) for nt 1D snapshots
               - array of the form (nt*ny,nx) for nt 2D snapshots
        
        Keywords:
        y
        :    y axis array for data of a form (ntny,nx) where the first ny rows are the fist snapshot
        
        data2
        :    for plotting additional 1-dimensional y(x) data on the 1D or 2D plot
            join in a list if several data-sets should be included like
            [y1 , y2] where y1,y2 are arrays of shape (nt,nx)
            
        data3
        :    for plotting additional vertical lines on the 1D or 2D plot
            can be either nt points for x(t)-data or one single float if time dependent
            join to lists if more data is plotted
             
        times
        :    times of the snapshots, to be shown in the title of the axes, if given
        
        i_start
        :    index of initial snapshot
        [x,y,z]log
        :    true: use logarithmic scale in [x,y,z]
        
        [x,y,z]lim
        :    give limits [x0,x1], ... for the specified axes    
        
        [x,y]label
        :    label for the [x,y] axes
        
        lstyle
        :    style (string or color specification) or list of styles to be used for the lines
            will be repeatet if too short
        ncont
        :    number of contours for the contour plot
        
        cmap
        :    color map for the contours
        
        fill
        :    if true, data lower than zlim[0] will be rendered at lowest color level
            if false, will be rendered white
            
        ext_link
        :    link an onther plotter object to the slider of the current one
        
        """
        #
        # general setup
        #
        if y!=None:
            if np.ndim(x)!=np.ndim(y):
                print('ERROR: x and y need to be both 1D or both 2D')
                sys.exit(1)        
            
            if (np.ndim(x)==2) and (x.shape!=y.shape):
                print('ERROR: if x and y are given as 2D arrays, shape(x) need to be equal to shape(y)')
                sys.exit(1)        
            
        if np.ndim(x)==1:
            nx = len(x)
        else:
            nx = x.shape[1]
            
        if y==None:
            nt = data.shape[0]
        else:
            if np.ndim(y)==1:
                ny = len(y)
            else:
                ny = x.shape[0]
            nt = data.shape[0]/ny
        #
        # some size checks
        #    
        if nx!=data.shape[1]:
            print('ERROR: number of x points does not match the number of columns of the data array')
            sys.exit(1)
        if times!=None:
            if nt != len(times):
                print('ERROR: len(times) does not match (number of rows)/ny of the data array')
                sys.exit(1)
        if y==None:
            i_max  = data.shape[0]-1
        else:
            i_max  = nt-1
        #
        # color scheme 
        #
        if cmap==None: cmap=plt.get_cmap('hot')
        #
        # convert data2 if necessary
        #
        if type(data2).__name__=='ndarray':
            data2 = [data2]
        #
        # convert data3 if necessary
        #
        if type(data3).__name__=='ndarray':
            data3 = [data3]
        #
        # convert to arrays
        #
        for i in np.arange(len(data3)):
            data3[i]=np.array(data3[i],ndmin=1)
        #
        # set limits
        #
        if xlim==None: xlim=[x.min(),x.max()]
        if ylim==None:
            if y==None:
                ylim=[data.min(),data.max()]
            else:
                ylim=[y.min(),y.max()]
        if zlim==None: zlim=[data.min(),data.max()]
        #
        # zlog cannot just be set, we need to convert the data
        #
        if zlog:
            data=np.log10(data)
            zlim=np.log10(zlim)
        #
        # add floor value
        #
        if fill:
            data = np.maximum(data,zlim[0])
        #
        # set number of contours
        #
        if ncont==None:
            ncont=zlim[-1]-zlim[0]+1
        #
        # set line styles
        #
        if type(lstyle).__name__!='list': lstyle=[lstyle]
        len_ls0 = len(lstyle)
        len_ls1 = len(data2)+len(data3)+1
        dummy = []
        for j in np.arange(len_ls1):
            dummy += [lstyle[np.mod(j,len_ls0)]]
        lstyle = dummy
        #
        # set up figure
        #
        plt.figure()
        #
        # ===============
        # INITIAL DRAWING
        # ===============
        #
        ax    = plt.subplot(111)
        plt.subplots_adjust(left=0.25, bottom=0.25)
        plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
        #
        # draw labels
        #
        if xlabel!='': plt.xlabel(xlabel)
        if ylabel!='': plt.ylabel(ylabel)
        if times!=None: ti=plt.title('%g'%times[i_start])
        #
        # set scales
        #
        if xlog: ax.set_xscale('log')
        if ylog: ax.set_yscale('log')
        #
        # plot the normal data
        #
        if y==None:
            #
            # line data
            #
            if type(lstyle[0]).__name__=='str':
                l, = ax.plot(x,data[i_start],lstyle[0],lw=2)
            else:
                l, = ax.plot(x,data[i_start],color=lstyle[0],lw=2)
        else:
            #
            # 2D data
            #
            l  = ax.contourf(x,y,data[i_start*ny+np.arange(ny),:],np.linspace(zlim[0],zlim[-1],ncont),cmap=cmap)
            clist = l.collections[:]
        #
        # plot additional line data
        #
        add_lines = []
        for j,d in enumerate(data2):
            if type(lstyle[j+1]).__name__=='str':
                l2, = ax.plot(x,d[i_start],lstyle[j+1],lw=2)
            else:
                l2, = ax.plot(x,d[i_start],color=lstyle[j+1],lw=2)
            add_lines+=[l2]
        #
        # plot additional vertical lines
        #
        add_lines2 = []
        for j,d in enumerate(data3):
            if type(lstyle[j+1]).__name__=='str':
                l3, = ax.plot(d[min(i_start,len(d)-1)]*np.ones(2),ax.get_ylim(),lstyle[j+1+len(data2)],lw=2)
            else:
                l3, = ax.plot(d[min(i_start,len(d)-1)]*np.ones(2),ax.get_ylim(),color=lstyle[j+1+len(data2)],lw=2)
            add_lines2+=[l3]
        #
        # ========
        # Make GUI
        # ========
        #
        #
        # make time slider
        #
        axcolor     = 'lightgoldenrodyellow'
        ax_time     = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        slider_time = Slider(ax_time, 'time', 0.0, i_max, valinit=i_start,valfmt='%i')
        self.slider = slider_time
        ax._widgets = [slider_time] # avoids garbage collection
        #
        # define slider update funcion
        #
        def update(val):
            i = int(np.floor(slider_time.val))
            if y==None:
                #
                # update line data
                #
                l.set_ydata(data[i])
            else:
                #
                # update 2D data
                #            
                while len(clist)!=0:
                    for col in clist:
                        ax.collections.remove(col)
                        clist.remove(col)
                dummy  = ax.contourf(x,y,data[i*ny+np.arange(ny),:],np.linspace(zlim[0],zlim[-1],ncont),cmap=cmap)
                for d in dummy.collections:
                    clist.append(d)
            #
            # update additional lines
            #
            for d,l2 in zip(data2,add_lines):
                l2.set_ydata(d[i])
            #
            # update additional vertical lines
            #
            for d,l3 in zip(data3,add_lines2):
                l3.set_xdata(d[min(i,len(d)-1)])
                l3.set_ydata(ax.get_ylim())
            #
            # update title
            #
            if times!=None: ti.set_text('%g'%times[i])
            #
            # update plot
            #
            plt.draw()
            #
            # update external plotter as well
            #
            if ext_link!=None: ext_link.slider.set_val(slider_time.val)
        slider_time.on_changed(update)
        #
        # set xlog button
        #
        ax_xlog = plt.axes([0.5, 0.025, 0.1, 0.04])
        button_xlog = Button(ax_xlog, 'xscale', color=axcolor, hovercolor='0.975')
        ax._widgets += [button_xlog] # avoids garbage collection
        def xlog_callback(event):
            if ax.get_xscale() == 'log':
                ax.set_xscale('linear')
            else:
                ax.set_xscale('log')
            plt.draw()
        button_xlog.on_clicked(xlog_callback)
        #
        # set ylog button
        #
        ax_ylog = plt.axes([0.6, 0.025, 0.1, 0.04])
        button_ylog = Button(ax_ylog, 'yscale', color=axcolor, hovercolor='0.975')
        ax._widgets += [button_ylog] # avoids garbage collection
        def ylog_callback(event):
            if ax.get_yscale() == 'log':
                ax.set_yscale('linear')
            else:
                ax.set_yscale('log')
            plt.draw()
        button_ylog.on_clicked(ylog_callback)
        #
        # plot button
        #
        ax_plotbutton = plt.axes([0.8, 0.025, 0.1, 0.04])
        button_plot = Button(ax_plotbutton, 'plot', color=axcolor, hovercolor='0.975')
        def plotbutton_callback(event,img_name=None,img_format='.pdf'):
            # ===================================================
            # this part is copied from above, replacing ax=>newax
            # and getting the snapshot index from the slider
            # ===================================================
            #
            newfig=plt.figure();
            newax    = plt.subplot(111);
            i        = int(np.floor(slider_time.val));
            plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]]);
            #
            # draw labels
            #
            if xlabel!='': plt.xlabel(xlabel)
            if ylabel!='': plt.ylabel(ylabel)
            if times!=None: ti=plt.title('%g'%times[i])
            #
            # set scales
            #
            if xlog: newax.set_xscale('log')
            if ylog: newax.set_yscale('log')
            #
            # plot the normal data
            #
            if y==None:
                #
                # line data
                #
                if type(lstyle[0]).__name__=='str':
                    l, = newax.plot(x,data[i],lstyle[0],lw=2)
                else:
                    l, = newax.plot(x,data[i],color=lstyle[0],lw=2)
            else:
                #
                # 2D data
                #
                l  = newax.contourf(x,y,data[i*ny+np.arange(ny),:],np.linspace(zlim[0],zlim[-1],ncont),cmap=cmap)
                clist = l.collections[:]
            #
            # plot additional line data
            #
            add_lines = []
            for j,d in enumerate(data2):
                if type(lstyle[j+1]).__name__=='str':
                    l2, = newax.plot(x,d[i],lstyle[j+1],lw=2)
                else:
                    l2, = newax.plot(x,d[i],color=lstyle[j+1],lw=2)
                add_lines+=[l2]
            #
            # plot additional vertical lines
            #
            add_lines2 = []
            for j,d in enumerate(data3):
                if type(lstyle[j+1]).__name__=='str':
                    l3, = newax.plot(d[min(i,len(d)-1)]*np.ones(2),newax.get_ylim(),lstyle[j+1+len(data2)],lw=2)
                else:
                    l3, = newax.plot(d[min(i,len(d)-1)]*np.ones(2),newax.get_ylim(),color=lstyle[j+1+len(data2)],lw=2)
                add_lines2+=[l3]
            #
            # =========================================
            # now set the limits as in the other figure
            # =========================================
            #
            newax.set_xlim(ax.get_xlim())
            newax.set_ylim(ax.get_ylim())
            newax.set_xscale(ax.get_xscale())
            newax.set_yscale(ax.get_yscale())
            #
            # =========================================
            # now set the limits as in the other figure
            # =========================================
            #
            if '.' not in img_format: img_format = '.'+img_format
            if img_name==None:
                j=0
                while os.path.isfile('figure_%03i%s'%(j,img_format)): j+=1
                img_name = 'figure_%03i%s'%(j,img_format)
            else:
                img_name = img_name.replace(img_format,'')+img_format
            plt.savefig(img_name)
            print('saved %s'%img_name)
            plt.close(newfig)
        button_plot.on_clicked(plotbutton_callback)
        ax._widgets += [button_plot] # avoids garbage collection
        #
        # plot button
        #
        ax_moviebutton = plt.axes([0.7, 0.025, 0.1, 0.04])
        button_movie = Button(ax_moviebutton, 'movie', color=axcolor, hovercolor='0.975')
        def moviebutton_callback(event):
            dirname    = 'movie_images_'+''.join(choice(ascii_letters) for x in range(5))
            img_format = '.png'
            #
            # create folder
            #
            if os.path.isdir(dirname):
                print('WARNING: %s folder already exists, please delete it first'%dirname)
                return
            else:
                os.mkdir(dirname)
            #
            # save all the images
            #
            i0 = int(np.floor(slider_time.val))
            for j,i in enumerate(np.arange(i0,nt)):
                slider_time.set_val(i)
                plotbutton_callback(None,img_name=dirname+os.sep+'img_%03i'%j, img_format=img_format);
            #
            # create the movie
            #
            moviename = 'movie.mp4'
            i_suffix  = 0
            dummy     = moviename
            while os.path.isfile(dummy):
                i_suffix += 1
                dummy     = moviename.replace('.', '_%03i.'%i_suffix)
            moviename = dummy
            ret=subprocess.call(['ffmpeg','-i',dirname+os.sep+'img_%03d'+img_format,'-c:v','libx264','-crf','20','-maxrate','400k','-pix_fmt','yuv420p','-bufsize','1835k',moviename]);
            if ret==0:
                #
                # delete the images & the folder
                #
                for j,i in enumerate(np.arange(i0,nt)):
                    os.remove(dirname+os.sep+'img_%03i%s'%(j,img_format))
                shutil.rmtree(dirname)
                print('*** Movie successfully created ***')
            else:
                print('WARNING: movie could not be produced, keeping images')
            #
            # reset slider
            #
            slider_time.set_val(i0)
        button_movie.on_clicked(moviebutton_callback)
        ax._widgets += [button_movie] # avoids garbage collection
        #
        # make ax current axes, so that it is easier to interact with
        #
        plt.axes(ax)
        #
        # GO
        #
        plt.draw()
