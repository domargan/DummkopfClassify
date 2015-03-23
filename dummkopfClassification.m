% Data must be like this:
% X -> data matrix where each row represents an observation, and each column represents a feature
% labels -> class labels (cell vector)
% features -> feature labels (cell vector)

% Number of observation points, classes and features
nObsv = length(X)
nClasses = length(unique(labels))
nFeatures = length(unique(features))


% Find indices of each class members individually 
labelsUnique = unique(labels);
classIdxUnique = {}
for i = 1:length(labelsUnique),
    classIdxUnique{i} = find(ismember(labels, labelsUnique(i)))
end
% length(classIdxUnique) == nClasses   % must be 1 


% Feature scaling to [0-1]
scaledX = (X - min(X(:))) ./ (max(X(:) - min(X(:))));
X = scaledX;


% Feature correlation
c = corr(X);
figure, imagesc(c), colorbar
set(gca, 'XTick', linspace(1, nFeatures, nFeatures))
set(gca, 'YTick', linspace(1, nFeatures, nFeatures))

[i, j] = find(c > 0.8);
correlated = [i, j];
correlated = correlated(find(arrayfun(@(i)length(unique(correlated(i, :))), 1:size(correlated, 1)) == size(correlated, 2)), :);
correlatedUnique = unique(sort(correlated, 2), 'rows');
clearvars i j;


% PCA
[pcaCoeff, XPca] = princomp(zscore(X));
boxplot(pcaCoeff, 'orientation', 'horizontal', 'labels', features)


% Raw data visualization
pairs = []
for i = 2:nFeatures, pairs = [pairs; [1 i]], end
h = figure;
for j = 1:(nFeatures - 1),
    x = pairs(j, 1);
    y = pairs(j, 2);
    subplot(ceil(sqrt(nFeatures)), floor(sqrt(nFeatures)), j);
    gscatter(X(:, x), X(:, y), labels);
    xlabel(features{x}, 'FontSize', 15);
    ylabel(features{y}, 'FontSize', 15);
end
clearvars i j;


% k-means clustering
kmeansIdxEuc = kmeans(X, nClasses, 'distance', 'sqEuclidean', 'replicates', 10);
[silhEuc, silhFigEuc] = silhouette(X, kmeansIdxEuc, 'sqEuclidean');
silhEuc = mean(silhEuc)

kmeansIdxCity = kmeans(X, nClasses, 'distance', 'cityblock', 'replicates', 10);
[silhCity, silhFigCity] = silhouette(X, kmeansIdxCity, 'cityblock');
silhCity = mean(silhCity)

kmeansIdxCosine = kmeans(X, nClasses, 'distance', 'cosine', 'replicates', 10);
[silhCosine, silhFigCosine] = silhouette(X, kmeansIdxCosine, 'cosine');
silhCosine = mean(silhCosine)

kmeansIdxCorr = kmeans(X, nClasses, 'distance', 'correlation', 'replicates', 10);
[silhCorr, silhFigCorr] = silhouette(X, kmeansIdxCorr, 'correlation');
silhCorr = mean(silhCorr)


% Hierarhical cluster tree
hierClustTree = linkage(X, 'complete');
clusters = cluster(hierClustTree, 'maxclust', nClasses);
for i = 1:nClasses, size(find(clusters == i)), end
scatter3(X(:, 1), X(:, 5), X(:, 7), 30, clusters, 'filled');
dendrogram(hierClustTree, 'colorthreshold', 'default');

 
% Split data to train and test
cp = cvpartition(labels, 'k', 5);
disp(cp)
trIdx = cp.training(1);
teIdx = cp.test(1);


% Classification trees
%t = classregtree(X(teIdx, :), labels(teIdx, :), 'names', features);
t = classregtree(X, labels, 'names', features);
view(t)

tree = ClassificationTree.fit(X(trIdx, :), labels(trIdx, :))
[treeClass, treeScore] = predict(tree, X(teIdx, :));

bad = ~strcmp(treeClass, labels(teIdx, :)); 
treeResubErr = sum(bad) / nObsv 

[treeResubCM, grpOrder] = confusionmat(labels(teIdx, :), treeClass)

treeStats = confusionmatStats(confusionmat(labels(teIdx, :), treeClass))

view(tree, 'mode', 'graph')

[falsePositiveTree, truePositiveTree, T, AucTree] = perfcurve(labels(teIdx, :), treeScore(:, 1), labels{1});
plot(falsePositiveTree, truePositiveTree, 'LineWidth', 5)
xlabel('False positive rate'); 
ylabel('True positive rate');
title('ROC');

predict(tree, mean(X))



% LDA & QDA
% Old way: ldaClass = classify(X(teIdx, :), X(trIdx, :), labels(trIdx, :));
% For QDA add: ClassificationDiscriminant.fit(X(trIdx, :), labels(trIdx,
% :),'discrimType', 'quadratic');
lda = ClassificationDiscriminant.fit(X(trIdx, :), labels(trIdx, :));
[ldaClass, ldaScore] = predict(lda, X(teIdx, :));

bad = ~strcmp(ldaClass, labels(teIdx, :)); 
ldaResubErr = sum(bad) / nObsv 

[ldaResubCM, grpOrder] = confusionmat(labels(teIdx, :), ldaClass)

ldaStats = confusionmatStats(confusionmat(labels(teIdx, :), ldaClass))

gscatter(X(:, 1), X(:, 9), labels);
hold on;
plot(X(bad, 1), X(bad, 9), 'kx');
hold off;

% Old way: ldaClassFun = @(xtrain, ytrain, xtest)(classify(xtest, xtrain, ytrain));
ldaClassFun = @(xtrain, ytrain, xtest)...
               (predict(ClassificationDiscriminant.fit(xtrain, ytrain), xtest));
ldaCvErr  = crossval('mcr', X, labels, 'predfun', ...
             ldaClassFun, 'partition', cp)

[falsePositiveLda, truePositiveLda, T, AucLda] = perfcurve(labels(teIdx, :), ldaScore(:, 1), labels{1});
plot(falsePositiveLda, truePositiveLda, 'LineWidth', 5)
xlabel('False positive rate'); 
ylabel('True positive rate');
title('ROC');

predict(lda, mean(X))

       
% Naive Bayes
nb = NaiveBayes.fit(X(trIdx, :), labels(trIdx, :), 'Distribution', 'kernel');
nbClass = predict(nb, X(teIdx, :));

bad = ~strcmp(nbClass, labels(teIdx, :));
nbResubErr = sum(bad) / nObsv

nbClassFun = @(xtrain, ytrain, xtest)...
               (predict(NaiveBayes.fit(xtrain, ytrain, 'Distribution', 'kernel'), xtest));
nbCvErr = crossval('mcr', X, labels,...
              'predfun', nbClassFun, 'partition', cp)
          
confusionmat(labels(teIdx, :), nbClass)

nbStats = confusionmatStats(confusionmat(labels(teIdx, :), nbClass))

pNb = posterior(nb,X(teIdx, :));
[falsePositiveNb, truePositiveNb, T, AucNb] = perfcurve(labels(teIdx, :), pNb(:, 1), labels{1});
plot(falsePositiveNb, truePositiveNb, 'LineWidth', 5)
xlabel('False positive rate'); 
ylabel('True positive rate');
title('ROC');

predict(nb, mean(X))	


% KNN
knn = ClassificationKNN.fit(X(trIdx, :), labels(trIdx, :), 'NumNeighbors', 5, 'Distance', 'cosine');
[knnClass, knnScore] = predict(knn, X(teIdx, :));

bad = ~strcmp(knnClass, labels(teIdx, :));
knnResubErr = sum(bad) / nObsv

knnClassFun = @(xtrain, ytrain, xtest)...
               (predict(ClassificationKNN.fit(xtrain, ytrain), xtest));
knnCvErr  = crossval('mcr', X, labels,...
              'predfun', knnClassFun, 'partition', cp)

knnStats = confusionmatStats(confusionmat(labels(teIdx, :), knnClass))

[falsePositiveKnn, truePositiveKnn, T, AucKnn] = perfcurve(labels(teIdx, :), knnScore(:, 1), labels{1});
plot(falsePositiveKnn, truePositiveKnn, 'LineWidth', 5)
xlabel('False positive rate'); 
ylabel('True positive rate');
title('ROC');

predict(knn, mean(X))


% SVM 
% Classify every possible class pairs
svmResubErrMat = [];
svmCvErrMat = [];
scvStatsMat = [];
for i = 1:length(classIdxUnique),
    for j = i:length(classIdxUnique),
        if i == j
            svmResubErrMat{i, j} = NaN,
            svmCvErrMat{i, j} = NaN,
            svmStatsMat{i, j} = NaN,
        else
            svmX = X([classIdxUnique{i}; classIdxUnique{j}], :);
            svmLabels = labels([classIdxUnique{i}; classIdxUnique{j}], :);
            svmCp = cvpartition(svmLabels, 'k', 5);
            svmTrIdx = svmCp.training(1);
            svmTeIdx = svmCp.test(1);
            nObsvSvm = size(svmX, 1);

            %svmStruct = svmtrain(svmX(svmTrIdx, [1 9]),svmLabels(svmTrIdx), 'ShowPlot', true);
            svmStruct = svmtrain(svmX(svmTrIdx, :), svmLabels(svmTrIdx, :));
            svmClass = svmclassify(svmStruct, svmX(svmTeIdx, :));

            svmBad = ~strcmp(svmClass, svmLabels(svmTeIdx, :));
            svmResubErr = sum(svmBad) / nObsvSvm;

            svmResubErrMat{i, j} = svmResubErr;
            svmResubErrMat{j, i} = svmResubErr;

            svmStats = confusionmatStats(svmLabels(svmTeIdx, :), svmClass);
            svmStatsMat{i, j} = svmStats;
            svmStatsMat{j, i} = svmStats;
    
            svmClassFun = @(xtrain, ytrain, xtest)...
                           (svmclassify(svmtrain(xtrain, ytrain), xtest));
            svmCVErr  = crossval('mcr', svmX, svmLabels,...
                          'predfun', svmClassFun, 'partition', svmCp)

            svmCvErrMat{i, j} = svmCVErr;
            svmCvErrMat{j, i} = svmCVErr;
        end
    end
end

svmResubErrMat = cell2mat(svmResubErrMat)
svmCvErrMat = cell2mat(svmCvErrMat)

figure, imagesc(svmResubErrMat), colorbar
set(gca, 'XTick', linspace(1, nClasses, nClasses))
set(gca, 'YTick', linspace(1, nClasses, nClasses))

figure, imagesc(svmCvErrMat), colorbar
set(gca, 'XTick', linspace(1, nClasses, nClasses))
set(gca, 'YTick', linspace(1, nClasses, nClasses))

svmclassify(svmStruct, mean(svmX))