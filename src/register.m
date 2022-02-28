
moving = rgb2gray(imread('tmp/im2f.png'));
fixed = rgb2gray(imread('tmp/im1f.png'));

[optimizer,metric] = imregconfig('multimodal');

tform = imregtform(moving,fixed,'rigid',optimizer,metric);
moving_reg = imwarp(moving,tform);
[fmin, fmax, cmin, cmax] = borders2delete(moving_reg);
moving_reg = moving_reg(fmin:fmax,cmin:cmax);
imwrite(moving_reg, 'tmp/im2f.png')

moving = imread('tmp/im2.png');

moving_reg = imwarp(moving,tform);
moving_reg = moving_reg(fmin:fmax,cmin:cmax,:);
imwrite(moving_reg, 'tmp/im2.png')



function [fmin, fmax, cmin, cmax] = borders2delete(moving_reg)

    suma = sum(moving_reg,2);
    
    cont = 1; while suma(cont)==0, cont = cont+1; end
    if cont>5
        fmin = cont - 5;
    else
        fmin = 1;
    end
    
    cont = size(moving_reg,1); while suma(cont)==0, cont = cont-1; end
    if cont<size(moving_reg,1)-5
        fmax = cont + 5;
    else
        fmax = size(moving_reg,1);
    end
    
    
    suma = sum(moving_reg,1);
    
    cont = 1; while suma(cont)==0, cont = cont+1; end
    if cont>5
        cmin = cont - 5;
    else
        cmin = 1;
    end
    
    cont = size(moving_reg,2); while suma(cont)==0, cont = cont-1; end
    if cont<size(moving_reg,2)-5
        cmax = cont + 5;
    else
        cmax = size(moving_reg,2);
    end
end    
    
    
    
    
    
    
    
    
    
    
