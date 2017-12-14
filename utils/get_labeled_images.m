clear all
addpath('/home/deeplearning/teera/LabelMeToolbox/main')
addpath('/home/deeplearning/teera/LabelMeToolbox/XMLtools')
addpath('/home/deeplearning/teera/LabelMeToolbox/utils')
addpath('/home/deeplearning/teera/LabelMeToolbox/querytools')
addpath('/home/deeplearning/teera/LabelMeToolbox/3Dtools')
addpath('/home/deeplearning/teera/LabelMeToolbox/imagemanipulation')

HOMEIMAGES = '/home/deeplearning/teera/bridge_dataset'; % you can set here your default folder
HOMEANNOTATIONS = '/home/deeplearning/teera/bridge_annotations/'; % you can set here your default folder
OP_ANNOTATIONS = '/home/deeplearning/teera/bridge_masks';

D = LMdatabase(HOMEANNOTATIONS);
relativearea = LMlabeledarea(D);
ndx = find(relativearea>.1);
D = D(ndx);
relativearea = relativearea(ndx);
Nimages = length(D);

for ndx = 1:Nimages
    annotation = D(ndx).annotation;
    [mask, class] = LMobjectmask(annotation, HOMEIMAGES);
    m = zeros(size(mask,1), size(mask,2), 3);
    class_name = {annotation.object.name};
    for c = 1:length(class_name)
        colored_mask = cat(3, mask(:,:,c), mask(:,:,c), mask(:,:,c));
        if strcmp(class_name{c}, "delamination")
            % yellow
            colored_mask(:,:,3)=0;
            m = m + colored_mask;
        elseif strcmp(class_name{c}, "rebar_exposure")
            % red
            colored_mask(:,:,2)=0;
            colored_mask(:,:,3)=0;
            for idx = 1:numel(m)
                if (colored_mask(idx) ~= 0)
                    m(idx) = colored_mask(idx);
                end
            end
        end
    end
    write_dir = fullfile(OP_ANNOTATIONS, D(ndx).annotation.folder, [D(ndx).annotation.filename(1:end-4) '.png']);
    imwrite(m, write_dir)
end

