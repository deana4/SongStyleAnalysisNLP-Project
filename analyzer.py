import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from pca import pca
import plotly.graph_objs as go
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d


class utils:
    colorList = ['blue', 'green', 'red', 'purple', 'yellow', 'cyan', 'orange', 'pink']

    @staticmethod
    def saveDict(dictToSave, filename):
        with open(f'{filename}.json', 'w') as file:
            json.dump(dictToSave, file, indent=4)

    @staticmethod
    def readDict(filename):
        with open(filename, 'r') as file:
            loadedDict = json.load(file)
        return loadedDict

    @staticmethod
    def plot3DGraphForAvgClassVector(x, y, z, labels, axises, title=""):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        plt.ion()

        # Create a scatter plot
        scatter = ax.scatter(x, y, z, c=np.arange(len(x)), cmap='tab20', s=75)

        # Create a legend using scatter.legend_elements
        legend_elements, _ = scatter.legend_elements()
        ax.legend(legend_elements, labels, title="Data Items", bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize='small')

        ax.set_xlabel(axises[0])
        ax.set_ylabel(axises[1])
        ax.set_zlabel(axises[2])
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot3DGraphByClasses(X, Y, Z, labels, axises, styleMeans, title=""):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(X)):
            ax.scatter(X[i], Y[i], Z[i], c=utils.colorList[i], label=labels[i])

        for i in range(len(styleMeans)):
            ax.scatter(styleMeans[i][0], styleMeans[i][1], styleMeans[i][2], c=utils.colorList[i], s=165)

        ax.set_title(title)
        ax.set_xlabel(axises[0])
        ax.set_ylabel(axises[1])
        ax.set_zlabel(axises[2])

        ax.legend()
        plt.show()

    @staticmethod
    def plot2DGraphByClasses(X, Y, labels, axises, styleMeans, title=""):
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot each class
        for i in range(len(X)):
            ax.scatter(X[i], Y[i], c=utils.colorList[i], label=labels[i])

        for i in range(len(styleMeans)):
            ax.scatter(styleMeans[i][0], styleMeans[i][1], c=utils.colorList[i], s=165)

        # Labels and title
        ax.set_title(title)
        ax.set_xlabel(axises[0])
        ax.set_ylabel(axises[1])

        # Legend
        ax.legend()

        # Show plot
        plt.show()

    @staticmethod
    def plotly3D(X, Y, Z, labels, axises, title=""):
        traces = []
        for i in range(len(X)):
            trace = go.Scatter3d(
                x=X[i],
                y=Y[i],
                z=Z[i],
                mode='markers',
                marker=dict(
                    size=5,
                    color=utils.colorList[i % len(utils.colorList)],
                ),
                name=labels[i]
            )
            traces.append(trace)

        layout = go.Layout(
            title=title,
            scene=dict(
                xaxis_title=axises[0],
                yaxis_title=axises[1],
                zaxis_title=axises[2]
            ),
            legend=dict(
                x=0,
                y=1
            )
        )

        # Create figure and show
        fig = go.Figure(data=traces, layout=layout)
        fig.show()


n2s = utils.readDict('LabelToStyle.json')
s2n = utils.readDict('StyleToLabel.json')


class NormalizationMethods:
    @staticmethod
    def min_max_scaling(series):
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(series), columns=series.columns)

    @staticmethod
    def z_score_normalization(series):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(series), columns=series.columns)

    @staticmethod
    def robust_scaling(series):
        scaler = RobustScaler()
        return pd.DataFrame(scaler.fit_transform(series), columns=series.columns)

    @staticmethod
    def log_transformation(series):
        return pd.DataFrame(np.log1p(series), columns=series.columns)

    @staticmethod
    def maxabs_scaling(series):
        scaler = MaxAbsScaler()
        return pd.DataFrame(scaler.fit_transform(series), columns=series.columns)

    @staticmethod
    def decimal_scaling(series):
        max_abs = abs(series).max()
        scaling_factor = 10 ** len(str(int(max_abs)))
        return series / scaling_factor

    @staticmethod
    def unit_vector_transformation(series):
        norm = np.linalg.norm(series)
        return series / norm if norm != 0 else series


class DataVisualizer:

    @staticmethod
    def revealPcaFeatures(data, dims=3):
        df = pd.concat(data, axis=0)
        df_labels = df.iloc[:, -1]  # This selects the last column
        unlabeledDf = df.iloc[:, :-1]  # This selects all but the last column
        print(unlabeledDf.columns)
        n_components = min(unlabeledDf.shape[0], dims)
        myPCA = pca(n_components=n_components)

        data_pca = myPCA.fit_transform(unlabeledDf)
        print(data_pca['topfeat'])

    @staticmethod
    def activatePCA(data, labels, dims=3):
        df = pd.concat(data, axis=0)
        print(f'number of samples: {df.shape[0]}, number of features: {df.shape[1]}')
        df_labels = df.iloc[:, -1]  # This selects the last column
        unlabeledDf = df.iloc[:, :-1]  # This selects all but the last column

        # print(df_labels)
        n_components = min(unlabeledDf.shape[0], dims)
        regPca = PCA(n_components=n_components)
        data_pca = regPca.fit_transform(unlabeledDf)

        DataVisualizer.plotGraph(data_pca, df_labels, labels, "PCA Projection", dims)

    @staticmethod
    def activateTSNE(data, labels, dims=2):
        # Combine data
        df = pd.concat(data, axis=0)
        print(f'number of samples: {df.shape[0]}, number of features: {df.shape[1]}')

        # Extract labels and features
        df_labels = df.iloc[:, -1]  # This selects the last column
        unlabeledDf = df.iloc[:, :-1]  # This selects all but the last column

        # Initialize t-SNE
        tsne = TSNE(n_components=dims, random_state=0)
        data_tsne = tsne.fit_transform(unlabeledDf)

        DataVisualizer.plotGraph(data_tsne, df_labels, labels, "T-SNE Projection", dims)

    @staticmethod
    def activateLDA(data, labels, dims=2):
        df = pd.concat(data, axis=0)
        print(f'number of samples: {df.shape[0]}, number of features: {df.shape[1]}')

        df_labels = df.iloc[:, -1]
        unlabeledDf = df.iloc[:, :-1]

        # n_components = min(unlabeledDf.shape[1], len(labels))
        lda = LinearDiscriminantAnalysis(n_components=dims)
        data_lda = lda.fit_transform(unlabeledDf, df_labels)
        DataVisualizer.plotGraph(data_lda, df_labels, labels, "2D LDA", dims)

    @staticmethod
    def activateIsomap(data, labels, dims=2):
        df = pd.concat(data, axis=0)
        print(f'number of samples: {df.shape[0]}, number of features: {df.shape[1]}')

        df_labels = df.iloc[:, -1]
        unlabeledDf = df.iloc[:, :-1]

        isomap = Isomap(n_components=dims)
        data_isomap = isomap.fit_transform(unlabeledDf)

        DataVisualizer.plotGraph(data_isomap, df_labels, labels, "Isomap Projection", dims)

    @staticmethod
    def plotGraph(data_pca, sampleLabels, labels, title="", dim=2):
        assert dim in [2, 3]
        cList = ['blue', 'green', 'red', 'purple', 'yellow', 'cyan', 'orange', 'pink']
        color_map = {int(s2n[label]): cList[i] for i, label in enumerate(labels)}
        colors = sampleLabels.map(color_map)

        # Create handles for the legend
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[int(s2n[label])], markersize=5) for label
            in labels]
        legend_labels = labels

        fig = plt.figure(figsize=(10, 7))
        ax = None
        if dim == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=colors, s=75)
        else:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=colors, s=75)

        ax.legend(legend_handles, legend_labels, title="Styles", bbox_to_anchor=(1.05, 1), loc='best',
                  fontsize='small')
        if dim == 3:
            title = title.replace('2D', '3D')
        ax.set_title(title)
        ax.set_xlabel('axis1')
        ax.set_ylabel('axis2')

        if dim == 3:
            ax.set_zlabel('axis3')

        plt.show()


class FeatureAnalyzer:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.path = path

    def editColumn(self, column_name, reverse=True, Percentage=False):
        assert reverse is True and Percentage is False or reverse is False and Percentage is True
        column = self.data[column_name]
        if reverse:
            column = 1 - column
        if Percentage:
            column = 100 - column

        self.data[column_name] = column

    def applyNormalizations(self, dirc='normalizedDataSets'):
        # methods = [NormalizationMethods.min_max_scaling, NormalizationMethods.z_score_normalization,
        #            NormalizationMethods.robust_scaling,
        #            NormalizationMethods.log_transformation, NormalizationMethods.maxabs_scaling,
        #            NormalizationMethods.decimal_scaling,
        #            NormalizationMethods.unit_vector_transformation]
        methods = [NormalizationMethods.min_max_scaling, NormalizationMethods.z_score_normalization,
                   NormalizationMethods.robust_scaling]

        for method in methods:
            print(method.__name__)
            normDf = self.normalizeData(method)
            self.saveToCsv(normDf, f'{dirc}/{method.__name__}.csv')

    def applyLabeling(self):
        mapping = {}
        newColumn = []
        for i, value in enumerate(self.data['Music Style']):
            if type(value) is not float:
                value = value.strip()
                if value.endswith('”},\r\n\u200e'):
                    value = value.removesuffix('”},\r\n\u200e')

                value = FeatureAnalyzer.normalizeString(value)

                if value not in mapping.keys() and value.strip() != "" and value is not None:
                    mapping[value] = value
                # print(i, value, type(value))
            else:
                value = "Other"
                mapping[value] = value

            newColumn.append(value)

        self.data['Music Style'] = newColumn

        # print(self.data['Music Style'])

        labels = 0
        for key in mapping.keys():
            mapping[key] = labels
            labels += 1

        inverted_dict = {value: key for key, value in mapping.items()}

        utils.saveDict(inverted_dict, 'LabelToStyle')
        utils.saveDict(mapping, 'StyleToLabel')

        self.data['MappedStyle'] = self.data['Music Style'].map(mapping)

        # self.saveToCsv(self.data, "datasetMappedStyles.csv")

    def normalizeData(self, func, addArtists=False):
        columns = self.data.columns[8:]
        ignoreColumns = ['releaseYear', 'MusicStyle', 'percentageOfTotalWordsToUnique', 'MappedStyle', 'BirthYear']
        # ignoreColumns = ['releaseYear',
        #                  'MusicStyle',
        #                  'wordCount',
        #                  'WordsRhymes',
        #                  'percentageOfTotalWordsToUnique',
        #                  'MappedStyle',
        #                  'BirthYear',
        #                  'sentimentScore',
        #                  'DiffPOS',
        #                  'numberOfBiGrams',
        #                  'numberOfTriGrams',
        #                  'readabilityMeasure', 'positiveWords', 'negativeWords',
        #                  'semantic_similarity', 'heBERT_sentiment']

        dataStyles = self.data['MappedStyle']

        features = []
        for column in columns:
            if column not in ignoreColumns:
                features.append(column)

        print(f'The features selected for the analysis are: {features}')
        featuresDf = self.data.copy()[features]

        if func is not None:
            featuresDf = func(featuresDf)

        if not addArtists:
            featuresDf['MappedStyle'] = dataStyles

        if addArtists:
            featuresDf['Artist'] = self.data['artist']

        return featuresDf

    def groupingByMusicStyle(self, groupList, func=None, avg=False, batchSize=32):
        df = self.normalizeData(func=func, addArtists=False)
        sumGroups = []
        groupNames = []
        grouped = df.groupby('MappedStyle')

        for label, group in grouped:
            currentStyle = n2s[str(label)]
            if currentStyle in groupList:
                print(f"Group {label}, {currentStyle}: {len(group)}")
                if avg:
                    # Calculate the average vector for the group
                    sumVector = group.mean(numeric_only=True).to_frame().T
                    sumVector['MappedStyle'] = label
                    sumGroups.append(sumVector)
                else:
                    # Create batches and calculate average for each batch
                    num_batches = int(np.ceil(len(group) / batchSize))
                    batches = np.array_split(group, num_batches)
                    batch_list = []

                    for batch in batches:
                        batchSum = batch.mean(numeric_only=True).to_frame().T
                        batchSum['MappedStyle'] = label
                        batch_list.append(batchSum)

                    startDF = pd.concat(batch_list, ignore_index=True)
                    sumGroups.append(startDF)

                groupNames.append(currentStyle)

        return sumGroups, groupNames

    def groupingByArtist(self, groupList, func=None, avg=False, batchSize=32):
        df = self.normalizeData(func=func, addArtists=True)
        sumGroups = []
        groupNames = []
        grouped = df.groupby('Artist')
        for label, group in grouped:
            if label in groupList:
                print(f"Group {label}: {len(group)}")
                if avg:
                    # Calculate the average vector for the group
                    sumVector = group.mean(numeric_only=True).to_frame().T
                    sumVector['Artist'] = label
                    sumGroups.append(sumVector)
                else:
                    # Create batches and calculate average for each batch
                    num_batches = int(np.ceil(len(group) / batchSize))
                    batches = np.array_split(group, num_batches)
                    batch_list = []

                    for batch in batches:
                        batchSum = batch.mean(numeric_only=True).to_frame().T
                        batchSum['Artist'] = label
                        batch_list.append(batchSum)

                    startDF = pd.concat(batch_list, ignore_index=True)
                    sumGroups.append(startDF)

                groupNames.append(label)

        return sumGroups, groupNames

    def SyntaxComplexity(self):
        X_features = []
        Y_features = []
        for row in self.data.itertuples():
            repWords = 1 / getattr(row, 'numberOfRepeatedWords') if getattr(row, 'numberOfRepeatedWords') else 0
            Y_features.append(
                1 / 6 * getattr(row, 'readabilityMeasure') + 1 / 6 * repWords
                + 1 / 6 * getattr(row, "DiffLemmas") + 1 / 6 * getattr(row, "DiffPOS") +
                1 / 6 * getattr(row, "trigramsEntropy") + 1 / 6 * getattr(row, "bigramsEntropy")
            )

            X_features.append(getattr(row, 'releaseYear'))

        return X_features, Y_features

    def songLength(self):
        X_features = []
        Y_features = []
        for row in self.data.itertuples():
            Y_features.append(
                0.8 * getattr(row, 'wordCount') + 0.2 * getattr(row, 'numberOfRepeatedWords'))
            X_features.append(getattr(row, 'releaseYear'))

        return X_features, Y_features

    def VocabRichness(self):
        X_features = []
        Y_features = []
        for row in self.data.itertuples():
            Y_features.append(
                1 / 10 * getattr(row, 'avgSimilarityMeasure') + 3 / 10 * getattr(row, 'RatioOfPOStoWords') +
                3 * (1 / getattr(row, 'numberOfRepeatedWords') if getattr(row,
                                                                          'numberOfRepeatedWords') != 0 else 0) + 1 / 10 * getattr(
                    row, 'averageSetWordLength') +
                1 / 10 * getattr(row, 'word_similarity_large')
            )

            X_features.append(getattr(row, 'releaseYear'))

        return X_features, Y_features

    @staticmethod
    def saveToCsv(data, filename):
        data.to_csv(filename, encoding='utf-8-sig')

    @staticmethod
    def normalizeString(s):
        elements = [element.strip() for element in s.split(',')]
        elements.sort()
        return ', '.join(elements)

    def analyseStyle(self, genre, axis=(), func=None, avg=False, batchSize=32, reveal=False, lda_dims=2, pca_dims=3):
        groupedData, retLabels = self.groupingByMusicStyle(groupList=genre,
                                                           func=func,
                                                           avg=avg,
                                                           batchSize=batchSize)

        groupMeans = []

        if reveal:
            DataVisualizer.revealPcaFeatures(groupedData, dims=pca_dims)
            DataVisualizer.activateLDA(groupedData, retLabels, dims=3)
            DataVisualizer.activateLDA(groupedData, retLabels, dims=2)
            return

        X, Y, Z = [], [], []

        for g in groupedData:
            X.append([*g[axis[0]].values])
            Y.append([*g[axis[1]].values])
            Z.append([*g[axis[2]].values])

        for x, y, z in zip(X, Y, Z):
            groupMeans.append([np.mean(x), np.mean(y), np.mean(z)])

        utils.plot3DGraphByClasses(X, Y, Z,
                                   labels=retLabels,
                                   axises=axis,
                                   styleMeans=groupMeans)

        utils.plot2DGraphByClasses(X, Y,
                                   labels=retLabels,
                                   axises=axis[0:-1],
                                   styleMeans=groupMeans)

        # Remove comment for 3D interactive graph below
        # utils.plotly3D(X, Y, Z, labels=styles, axises=axis)

    def analyseArtist(self, artists, axis=(), func=None, avg=False, batchSize=32, reveal=False):
        groupedData, retLabels = self.groupingByArtist(groupList=artists,
                                                       func=func,
                                                       avg=avg,
                                                       batchSize=batchSize)

        groupMeans = []

        if reveal:
            DataVisualizer.revealPcaFeatures(groupedData, dims=3)
            # DataVisualizer.activateLDA(groupedData, retLabels, dims=3)
            # DataVisualizer.activateLDA(groupedData, retLabels, dims=2)
            return

        X, Y, Z = [], [], []

        for g in groupedData:
            X.append([*g[axis[0]].values])
            Y.append([*g[axis[1]].values])
            Z.append([*g[axis[2]].values])

        for x, y, z in zip(X, Y, Z):
            groupMeans.append([np.mean(x), np.mean(y), np.mean(z)])

        utils.plot3DGraphByClasses(X, Y, Z,
                                   labels=retLabels,
                                   axises=axis,
                                   styleMeans=groupMeans)

        utils.plot2DGraphByClasses(X, Y,
                                   labels=retLabels,
                                   axises=axis[0:-1],
                                   styleMeans=groupMeans)

        # Remove comment for 3D interactive graph below
        # utils.plotly3D(X, Y, Z, labels=styles, axises=axis)


def plot_data(X, Y, X_title, Y_title, plot_title="Data Plot"):
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, marker='o', linestyle='-', color='blue')
    plt.xlabel(X_title)
    plt.ylabel(Y_title)
    plt.title(plot_title)
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate X-axis labels if needed
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()


def activeStyleAnalytics(fa):
    # pcaModule = PcaOperator()
    # fa.applyNormalizations()
    # fa.applyLabeling()

    axes = ["bigramsEntropy", "trigramsEntropy", "word_similarity_large", "RatioOfPOStoWords"]
    styles = ["Pop", "Rock"]
    fa.analyseStyle(genre=styles,
                    axis=axes,
                    func=NormalizationMethods.z_score_normalization,
                    avg=False,
                    batchSize=128,
                    reveal=False)

    # -------------------------------------------------------------------------------- #

    # axes = ["numberOfRepeatedWords", "percentageOfRepeatedWords", "average_word_frequency"]
    # Axes for vocab richness
    # axes = ["average_word_frequency", "avg_word_similarity_hebrew", "ratioOfTotalWordsToUnique"]
    # axes = ["RatioOfPOStoWords", "avg_word_similarity_hebrew", "avg_word_similarity_english"]
    axes = ["DiffLemmas", "trigramsEntropy", "RatioOfPOStoWords"]
    styles = ["Mizrahi", "Pop"]
    fa.analyseStyle(genre=styles,
                    axis=axes,
                    func=NormalizationMethods.min_max_scaling,
                    avg=False,
                    batchSize=128,
                    reveal=False)

    # -------------------------------------------------------------------------------- #

    axes = ["numberOfRepeatedWords", "avg_word_similarity_hebrew", "average_word_frequency"]
    styles = ["Metal, Rock", "Pop, R&B"]
    fa.analyseStyle(genre=styles,
                    axis=axes,
                    func=NormalizationMethods.min_max_scaling,
                    avg=False,
                    batchSize=4,
                    reveal=False)

    # -------------------------------------------------------------------------------- #

    axes = ["AvgUniqueness", "avg_word_similarity_hebrew", "avg_word_similarity_english"]
    styles = ["Mizrahi", "Rock"]
    fa.analyseStyle(genre=styles,
                    axis=axes,
                    func=NormalizationMethods.min_max_scaling,
                    avg=False,
                    batchSize=64,
                    reveal=False)


def activeAnalyticsOverYears():
    X, Y = fa.SyntaxComplexity()

    df = pd.DataFrame({'Year': X, 'Value': Y})
    df['Year'] = df['Year'] - (df['Year'] % 2)
    df = df.sort_values('Year')
    # df_avg = df.groupby('Year', as_index=False)['Value'].mean()
    groupedDF = pd.DataFrame(df.groupby('Year', as_index=False)['Value'].median())
    newValues = gaussian_filter1d(groupedDF.values, 10)
    newValues = (newValues - newValues.mean()) / newValues.std()
    newValues = newValues + abs(newValues.min())
    groupedDF['Value'] = newValues
    plot_data(groupedDF['Year'], groupedDF['Value'], "Years", "Syntax Complexity Level")

    X, Y = fa.songLength()

    df2 = pd.DataFrame({'Year': X, 'Value': Y})
    df2['Year'] = df2['Year'] - (df2['Year'] % 4)
    df_avg2 = df2.groupby('Year', as_index=False)['Value'].median()
    print(df_avg2)
    plot_data(df_avg2['Year'], df_avg2['Value'], "Years", "Length of the songs and vocab richness over the years")

    X, Y = fa.VocabRichness()

    df2 = pd.DataFrame({'Year': X, 'Value': Y})
    df2['Year'] = df2['Year'] - (df2['Year'] % 4)
    df_avg2 = df2.groupby('Year', as_index=False)['Value'].median()
    print(df_avg2)
    plot_data(df_avg2['Year'], df_avg2['Value'], "Years", "Vocab richness over the years")


def activeArtistAnalytics(fa):
    axes = ["theUniquenessLvlOfTheRepeatedSongs", "AvgUniqueness", "uniqueWords"]
    artist = ["יהורם גאון", "עוזי חיטמן", "סאבלימינל", "סינרגיה", "אייל גולן", "עומר אדם"]
    fa.analyseArtist(artists=artist,
                     axis=axes,
                     func=NormalizationMethods.z_score_normalization,
                     avg=False,
                     batchSize=32,
                     reveal=False)

    axes = ['uniqueWords', 'numberOfBiGrams', 'WordsRhymes']
    styles = ["Pop", "Mizrahi", "Rock", "Hip-Hop"]
    fa.analyseStyle(genre=styles,
                    axis=axes,
                    func=NormalizationMethods.min_max_scaling,
                    avg=False,
                    batchSize=48,
                    reveal=False)


def activeSongStylePerYear(year_gap=5, csvPath='yearAndStyle.csv'):
    df = pd.read_csv(csvPath)

    if 'releaseYear' not in df.columns or 'MappedStyle' not in df.columns:
        raise ValueError("CSV file must contain 'releaseYear' and 'MappedStyle' columns")

    df['year_bucket'] = (df['releaseYear'] // year_gap) * year_gap

    unique_styles_per_year = df.groupby('year_bucket')['MappedStyle'].nunique()
    print(unique_styles_per_year)

    unique_styles_per_year.plot(kind='bar', figsize=(12, 8))
    plt.title(f'Number of Different Song Styles per Year (Grouped by {year_gap} Year{"s" if year_gap > 1 else ""})')
    plt.xlabel('Year')
    plt.ylabel('Number of Different Styles')
    plt.show()


if __name__ == "__main__":
    # fa = FeatureAnalyser("FixedLatestDataset.csv")
    fa = FeatureAnalyzer("datasetMappedStyles_final.csv")
    artistFromData = list(set(fa.data['artist']))

    # activeAnalyticsOverYears()
    # activeStyleAnalytics(fa)
    # activeArtistAnalytics(fa)
    #activeSongStylePerYear(4)
