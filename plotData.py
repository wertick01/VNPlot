import neutcurve
import matplotlib.pyplot as plt

class PlotData(object):

    def __init__(self, params):
        self.params = params

    def createHillCurvePlot(
        self,
        parsed_dataframe, 
        ):
        fits = neutcurve.CurveFits(
            parsed_dataframe, 
            fix_slope_first=False,
            allow_reps_unequal_conc=True,
            fixbottom=0,
            fixtop=100,
            infectivity_or_neutralized=self.params['infectivity_or_neutralized'],
            )
        fit_params = fits.fitParams(average_only=self.params['average_only'], no_average=self.params['no_average'], ic50_error="fit_stdev")
        fig, axs = fits.plotSera(
            xlabel="", 
            ylabel="", 
            ncol=self.params['ncol'],
            max_viruses_per_subplot=len(set(parsed_dataframe['virus'].values)),
            virus_to_color_marker = self.params['markers_colors_dict'],
            markersize = self.params['marker_size_px'],
            linewidth=self.params['marker_line_px'],
            legendtitle='',
            orderlegend=None,
            titlesize=self.params['subplot_title_fontsize'],
            labelsize=25,
            ticksize=20,
            legendfontsize=0,
            sharex=False,
            sharey=False,
            fix_lims=self.params['fix_lims'],
        )

        handles, labels = [], []

        for ax in axs:
            for i in ax:
                i.set_xticklabels(i.get_xticklabels(), fontdict={'family': self.params['font'], 'size': self.params['ylabel_fontsize']})
                i.xaxis.set_tick_params(labelsize=self.params['xlabel_fontsize'])
                i.grid(True, color='gray', linestyle='-', linewidth=2, alpha=0.5)
                i.set_facecolor('white')
                i.set_title(i.get_title(), fontdict={'family': self.params['font'], 'size': self.params['subplot_title_fontsize'], 'fontweight': 'bold'})
                pos = i.get_position()
                width = pos.width * 1.1
                i.set_position([pos.x0, pos.y0, width, pos.height])
                i.set_xlabel(self.params['xtitle'], labelpad=self.params['xlabel_pad'], fontdict={'family': self.params['font'], 'size': self.params['xlabel_title_fontsize']})
                i.set_ylabel(self.params['ytitle'], labelpad=self.params['ylabel_pad'], fontdict={'family': self.params['font'], 'size': self.params['ylabel_title_fontsize']})
                handles.append(i.get_legend_handles_labels()[0])
                labels.append(i.get_legend_handles_labels()[1])

                lines = i.get_lines()
                for j in range(len(lines)):
                    lines[j].set_markeredgecolor('black')
                    lines[j].set_markeredgewidth(1)

                i.legend([], [], frameon=False) 

                i.xaxis.set_label_coords(self.params['xaxis_coords'][0], self.params['xaxis_coords'][1])
                i.yaxis.set_label_coords(self.params['yaxis_coords'][0], self.params['yaxis_coords'][1])

        fig.subplots_adjust(wspace=self.params['wspace'], hspace=self.params['hspace'], bottom=self.params['bottom_padding'])

        return fig, axs, handles, labels, fit_params
    
    # Функция для удаления строк, где комбинация serum:virus равна False
    def filter_dataframe(self, parsed_dataframe, serum_virus_dict):
        filtered_df = parsed_dataframe[parsed_dataframe.apply(lambda row: serum_virus_dict.get(row['serum'], {}).get(row['virus'], False), axis=1)]
        return filtered_df
    
    def scale_to_percentage(self, arr):
        # Проверка, что минимальное значение меньше максимального
        min_val, max_val = min(arr), max(arr)
        for i in range(len(arr)):
            arr[i] = arr[i] - min_val
        percent = (max_val - min_val) / 100
        for i in range(len(arr)):
            arr[i] = arr[i] / percent

        return arr
    
    def plotData(
            self,
            parsed_dataframe,
            ):
        
        parsed_dataframe = self.filter_dataframe(parsed_dataframe, self.params['serum_virus_dict'])
        
        if self.params['scale']:
            for virus in set(parsed_dataframe['serum']):
                for serum in set(parsed_dataframe[parsed_dataframe['serum'] == virus]['virus']):
                    for replicate in set(parsed_dataframe[(parsed_dataframe['serum'] == virus) & (parsed_dataframe['virus'] == serum)].replicate):
                        modified = self.scale_to_percentage(parsed_dataframe[(parsed_dataframe['serum'] == virus) & (parsed_dataframe['virus'] == serum) & (parsed_dataframe['replicate'] == replicate)]["fraction infectivity"].values)
                        i = 0
                        for idx in parsed_dataframe[(parsed_dataframe['serum'] == virus) & (parsed_dataframe['virus'] == serum) & (parsed_dataframe['replicate'] == replicate)].index:
                            parsed_dataframe.loc[idx, "fraction infectivity"] = modified[i]
                            i += 1

        if self.params['reverse']:
            parsed_dataframe['fraction infectivity'] = 100 - parsed_dataframe['fraction infectivity']
            parsed_dataframe.loc[parsed_dataframe[parsed_dataframe['fraction infectivity'] < 0].index, 'fraction infectivity'] = 0
        
        fig, ax, handles, labels, fit_params = self.createHillCurvePlot(parsed_dataframe)
        fig.set_size_inches(self.params['fheight'], self.params['fwidth'])
        title = fig.suptitle(self.params['title'], fontdict={'family': self.params['font']}, fontsize=self.params['title_fontsize'], fontweight='bold', y=1.02)

        # Создание общей легенды для всех подграфиков
        handles = []
        labels = []
        for name, (color, marker) in self.params['markers_colors_dict'].items():
            handle = plt.Line2D([0, 0], [0, 0], marker=marker, color=color, linestyle='')
            handles.append(handle)
            labels.append(name)

        # Изменение параметров легенды
        legend = fig.legend(
            handles, labels, 
            loc='lower center', 
            ncol=self.params['legend_ncols'], 
            markerscale=self.params['marker_size'], 
            prop={'family': self.params['font'], 'size': self.params['legend_fontsize']},
            borderaxespad=self.params['legend_borderaxespad'])
        legend.get_frame().set_linewidth(0.0)  # Убираем рамку вокруг легенды

        # Добавление обводки к маркерам легенды
        for handle in legend.legendHandles:
            handle.set_markeredgecolor('black')  # Задаем цвет обводки
            handle.set_markeredgewidth(self.params['marker_line_width'])  # Задаем толщину обводки

        # Изменение расстояния от заголовка до графиков
        fig.subplots_adjust(top=self.params['title_padding'])  # Задаем нужное расстояние с помощью параметра top
        title.set_position((0.5, 1.02))

        return fig, ax, fit_params