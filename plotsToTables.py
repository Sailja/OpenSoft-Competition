import wx
import os
import pyPdf 
import pickle

from wand.image import Image
def split(filename, filewithpath):
        images = []
        print filename
        
        pdf_im = pyPdf.PdfFileReader(file(filewithpath, "rb"))
        npage = pdf_im.getNumPages()
        for p in range(npage):
                #im = PythonMagick.Image(str(filename) + '['+ str(p) +']')
                im=Image(filename=str(filewithpath) + '['+ str(p) +']',resolution=500)
      
                im.save(filename="images/im" + str(p)+ '.jpg')
               
                images.append("images/im" + str(p)+ '.jpg')
       
        return images
class MainWindow(wx.Frame):
    
    def __init__(self, parent, title):
        self.dirname = ''
        super(MainWindow, self).__init__(parent, title=title, size=(650, 650))
        self.CreateStatusBar()  # A Statusbar in the bottom of the window

        self.InitGUI()
        self.Centre()
        self.Show()

    def InitGUI(self):

        # MENUS
        fileMenu = wx.Menu()  # create menu for file
       
        helpMenu = wx.Menu()  # create a menu for help

        # FILE MENU
        menuOpen = fileMenu.Append(wx.ID_OPEN,
                        "&Open", " Open a file to edit")  # add open to File
        menuExit = fileMenu.Append(wx.ID_EXIT, "E&xit",
                                " Terminate the program")  # add exit to File

      
        # HELP MENU
        menuAbout = helpMenu.Append(wx.ID_ABOUT, "&About",
                                "About this program")  # add about menu item

        # MENUBAR
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")  # Adding the "filemenu" to the MenuBar
       
        menuBar.Append(helpMenu, "&Help")
        self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.

       
        # MENU EVENTS
        self.Bind(wx.EVT_MENU, self.OnOpen, menuOpen)
        self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)
        self.Bind(wx.EVT_MENU, self.OnExit, menuExit)

       

        # PANEL
        panel = wx.Panel(self)

        panel.SetBackgroundColour('#ededed')
        vBox = wx.BoxSizer(wx.VERTICAL)

        hBox = wx.BoxSizer(wx.HORIZONTAL)
        hBox.Add(wx.StaticText(panel, label="File:"), flag=wx.TOP, border=3)
        hBox.Add(wx.TextCtrl(panel), 1, flag=wx.LEFT, border=10)
        Open = wx.Button(panel, -1, "Open")
        self.Bind(wx.EVT_BUTTON, self.OnOpen,Open)
        hBox.Add(Open, 0, flag=wx.LEFT, border=10)
        printBtn = wx.Button(panel, -1, "Print tables")
        self.Bind(wx.EVT_BUTTON, self.printBtnClick,printBtn)
        hBox.Add(printBtn, 0, flag=wx.LEFT, border=10)

        vBox.Add(hBox, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)
        vBox.Add((-1, 10))

        
        
        hBox2 = wx.BoxSizer(wx.HORIZONTAL)
        
        hBox2.Add(wx.StaticText(panel, label="Tables:"))
        vBox.Add(hBox2, flag=wx.LEFT, border=10)
        vBox.Add((-1, 5))

        hBox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.urlFld = wx.TextCtrl(panel, style=wx.TE_MULTILINE, size=(-1, 495))
        hBox3.Add(self.urlFld, 1, flag=wx.EXPAND)
        vBox.Add(hBox3, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)


        panel.SetSizer(vBox)

    # GUI EVENTS
    def printBtnClick(self, e):
        self.urlFld.SetValue("No tables to display")
        #Add code here

    # MENU ITEM EVENTS
    def OnAbout(self, e):
        dlg = wx.MessageDialog(self, "How to operate",
                               "Select the pdf to work on", wx.OK)  # create a dialog (dlg) box to display the message, and ok button
        dlg.ShowModal()  # show the dialog box, modal means cannot do anything on the program until clicks ok or cancel
        dlg.Destroy()  # destroy the dialog box when its not needed

    def OnExit(self, e):
        self.Close(True)  # on menu item select, close the app frame.

    def OnOpen(self, e):
        self.file_open()
        e.Skip()

    

    def file_open(self):  # 9
        with wx.FileDialog(self, "Choose a file to open", self.dirname,
                           "", "*.*" , wx.OPEN) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                directory, filename = dlg.GetDirectory(), dlg.GetFilename()
                filewithpath = ('/'.join((directory, filename)))
    
                images = []
                if filename.endswith("pdf"):
                    images = split(filename,filewithpath)
                else:
                    if filename.endswith("jpg") or filename.endswith("png"):
                        images.append(str(filewithpath))
                        print str(filewithpath)
                        bashCommand = "python backup.py "+str(filewithpath)
                        os.system(bashCommand)
                        bashCommand = "python listToCSV.py"
                        os.system(bashCommand)
                        fname = pickle.load( open( "filelist.p", "rb" ) )
                        for f in fname:
                            bashCommand = "libreoffice "+f
                            os.system(bashCommand)
                    else:
                        print "Invalid file type"
                # print images
                return images

                    
app = wx.App(False)  # creates a new app
MainWindow(None, "Plots to Tables")  # give the frame a title
app.MainLoop()  # start the apps mainloop which handles app events
