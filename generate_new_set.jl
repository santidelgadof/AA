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
    
    dirNeg = "Bbox/No-enemigos/";
    dirPos = "Bbox/Enemigos/";

    for file in readdir(dirNeg)
        image = load("$dirNeg$file");
        image = imresize(image, ratio=1/2);
        #save image
        save("Bbox/No-enemigos_halfed/$file", image);

    end

    for file in readdir(dirPos)
        image = load("$dirPos$file");
        image = imresize(image, ratio=1/2);
        #save image
        save("Bbox/Enemigos_halfed/$file", image);
    end
    
end

loadImages();