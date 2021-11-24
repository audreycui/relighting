import argparse, os, shutil, json


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils import easydict
from easydict import EasyDict
from xml.etree import ElementTree as et
from collections import defaultdict

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--model', choices=['alexnet', 'vgg16', 'resnet152', 'progan'],
            default='alexnet')
    aa('--dataset', choices=['places', 'broden',
            'church', 'kitchen', 'livingroom'],
            default='places')
    aa('--seg', choices=['net', 'netp', 'netq', 'netpq',
            'netpqc', 'netpqxc', 'human'],
            default='net')
    aa('--layers', nargs='+')
    aa('--quantile', type=float, default=0.005)
    aa('--miniou', type=float, default=0.025)
    args = parser.parse_args()
    return args

def main():
    args = parseargs()
    threshold_iou = args.miniou
    layer_report = {}
    qdir = '-%d' % (args.quantile * 1000) if args.quantile != 0.005 else ''
    for layer in args.layers:
        input_filename = 'results/%s-%s-%s-%s%s/report.json' % (
                args.model, args.dataset, args.seg, layer, qdir) 
        with open(input_filename) as f:
            layer_report[layer] = EasyDict(json.load(f))
    # Now assemble the data needed for the graph
    # (Layername, [(catname, [unitcount, unitcount, unitcount]), (catname..)
    cat_order = ['object', 'part', 'material', 'color']
    graph_data = []
    for layer in args.layers:
        layer_data = []
        catmap = defaultdict(lambda: defaultdict(int))
        units = layer_report[layer].get('units',
                layer_report[layer].get('images', None)) # old format
        for unitrec in units:
            if unitrec.iou is None or unitrec.iou < threshold_iou:
                continue
            catmap[unitrec.cat][unitrec.label] += 1
        for cat in cat_order:
            if cat not in catmap:
                continue
            # For this graph we do not need labels
            cat_data = list(catmap[cat].values())
            cat_data.sort(key=lambda x: -x)
            layer_data.append((cat, cat_data))
        graph_data.append((layer, layer_data))
    # Now make the actual graph
    largest_layer = max(sum(len(cat_data)
            for cat, cat_data in layer_data)
            for layer, layer_data in graph_data)
    layer_height = 14
    layer_gap = 2
    barwidth = 3
    bargap = 0
    leftmargin = 48
    margin = 8
    svgwidth = largest_layer * (barwidth + bargap) + margin + leftmargin
    svgheight = ((layer_height + layer_gap) * len(args.layers) - layer_gap +
            2 * margin)
    textsize = 10

    # create an SVG XML element
    svg = et.Element('svg', width=str(svgwidth), height=str(svgheight),
            version='1.1', xmlns='http://www.w3.org/2000/svg')

    # Draw big category background rectangles
    y = margin
    for layer, layer_data in graph_data:
        et.SubElement(svg, 'text', x='0', y='0',
            style=('font-family:sans-serif;font-size:%dpx;' +
                'text-anchor:end;alignment-baseline:hanging;' +
                'transform:translate(%dpx, %dpx);') %
                (textsize, leftmargin - 4, y + (layer_height - textsize) / 2)
            ).text = str(layer)
        barmax = max(max(cat_data) if len(cat_data) else 1
                for cat, cat_data in layer_data) if len(layer_data) else 1
        barscale = float(layer_height) / barmax
        x = leftmargin
        for cat, cat_data in layer_data:
            catwidth = len(cat_data) * (barwidth + bargap)
            et.SubElement(svg, 'rect',
                    x=str(x), y=str(y),
                    width=str(catwidth),
                    height=str(layer_height),
                    fill=cat_palette[cat][1])
            for bar in cat_data:
                barheight = barscale * bar
                et.SubElement(svg, 'rect',
                        x=str(x), y=str(y + layer_height - barheight),
                        width=str(barwidth),
                        height=str(barheight),
                        fill=cat_palette[cat][0])
                x += barwidth + bargap
        y += layer_height + layer_gap

    # Output - this is the bare svg.
    result = et.tostring(svg).decode('utf-8')
    # Now add the file header.
    result = ''.join([
            '<?xml version=\"1.0\" standalone=\"no\"?>\n',
            '<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n',
            '\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n',
            result])
    output_filename = 'results/%s-%s-%s%s/multilayer-%d.svg' % (
                args.model, args.dataset, args.seg, qdir, args.miniou * 1000)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    print('writing to %s' % output_filename)
    with open(output_filename, 'w') as f:
        f.write(result)

cat_palette = {
    'object':   ('#4B4CBF', '#B6B6F2'),
    'part':     ('#55B05B', '#B6F2BA'),
    'material': ('#50BDAC', '#A5E5DB'),
    'texture':  ('#81C679', '#C0FF9B'),
    'color':    ('#F0883B', '#F2CFB6'),
    'other1':   ('#D4CF24', '#F2F1B6'),
    'other2':   ('#D92E2B', '#F2B6B6'),
    'other3':   ('#AB6BC6', '#CFAAFF')
}

if __name__ == '__main__':
    main()