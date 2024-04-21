using Images, ImageFiltering, ImageView, ImageMorphology, ColorTypes, ImageDraw, Colors
using Flux
using Statistics
using FileIO;
using DelimitedFiles;
using DataFrames
using Flux.Losses
using Random
using Random:seed!
using Plots
using Clustering
using ScikitLearn
using StatsBase
using ImageFeatures
using ImageTransformations

function loadImages()
    
    dirNeg = "Frames/binaries0/";
    dirPos = "Frames/binaries1/";

    for file in readdir(dirNeg)
        image = load("$dirNeg$file");
        image = imresize(image, ratio=1/4);
        #save image
        save("Frames/binaries0_4/$file", image);

    end

    for file in readdir(dirPos)
        image = load("$dirPos$file");
        image = imresize(image, ratio=1/4);
        #save image
        save("Frames/binaries1_4/$file", image);
    end
    
end

loadImages();