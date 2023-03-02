import torch
import visdom

vis=visdom.Visdom(env='first')
vis.text('fisrt visdom',win='text1')
vis.text('hello world',win='text1',append=True)

for i in range(100):
    vis.line(X=torch.tensor([i]),Y=torch.tensor([10*i+3]),win='loss',opts={'title':'y=2x'},update='append')

vis.image(torch.randn(3,256,256),win='image')