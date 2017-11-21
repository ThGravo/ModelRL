import matplotlib
gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
print ("I will test for", gui_env)
for gui in gui_env:
    print ("testing", gui)
    try:
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        print ("    ",gui, "Is Available")
        plt.plot([1.5,2.0,2.5])
        fig = plt.gcf()
        fig.suptitle(gui)
        plt.show()
        print ("Using ..... ",matplotlib.get_backend())
    except:
        print ("    ",gui, "Not found")