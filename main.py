import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D

def add_interactivity(fig, ax, G, nodes_data, node_size=300):
    coord_to_node_ids = {}
    for node_id, data in G.nodes(data=True):
        x = data.get('x_visual')
        y = data.get('y_visual')
        if pd.isna(x) or pd.isna(y): continue
        coord = (x, y)
        if coord not in coord_to_node_ids:
            coord_to_node_ids[coord] = []
        coord_to_node_ids[coord].append(node_id)

    hover_scatter = ax.scatter([], [], s=node_size * 1.5, facecolors='white', 
                               edgecolors='black', linewidths=2, zorder=20, visible=False)
    
    current_hover_coord = [None]
    info_legend = [None]
    main_legend = ax.get_legend()
    if main_legend:
        ax.add_artist(main_legend)

    def get_closest_node(event_x, event_y, threshold=100):
        if event_x is None or event_y is None: return None
        min_dist = float('inf')
        closest_coord = None
        for cx, cy in nodes_data.keys():
            dist = np.hypot(cx - event_x, cy - event_y)
            if dist < min_dist:
                min_dist = dist
                closest_coord = (cx, cy)
        if min_dist < threshold: 
            return closest_coord
        return None

    def on_hover(event):
        if event.inaxes != ax: return
        coord = get_closest_node(event.xdata, event.ydata)
        if coord != current_hover_coord[0]:
            current_hover_coord[0] = coord
            if coord:
                hover_scatter.set_offsets([coord])
                hover_scatter.set_visible(True)
            else:
                hover_scatter.set_visible(False)
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax: return
        coord = get_closest_node(event.xdata, event.ydata)
        
        if info_legend[0] is not None:
            try:
                info_legend[0].remove()
            except:
                pass
            info_legend[0] = None

        if coord:
            node_ids = coord_to_node_ids.get(coord, [])
            info_elements = []
            for nid in node_ids:
                data = G.nodes[nid]
                station_name = data.get('station_name', nid)
                info_elements.append(Line2D([0], [0], marker='none', linestyle='none', label=f"Station Name: {station_name}"))
                info_elements.append(Line2D([0], [0], marker='none', linestyle='none', label=f"Station Code: {nid}"))
                for key, value in data.items():
                    if pd.notna(value) and key not in ['x_visual', 'y_visual', 'station_name', 'color','x-text','y-text','ha','va','next_node','edge_type','edge_color','curve_side','curve_smoothness','curve_bulge','linewidth']:
                        info_elements.append(Line2D([0], [0], marker='none', linestyle='none', label=f"{key.capitalize()}: {value}"))
                info_elements.append(Line2D([0], [0], marker='none', linestyle='none', label="-" * 20))
            if info_elements:
                info_elements.pop()

            info_legend[0] = ax.legend(handles=info_elements, loc='upper right', title="Station Info", 
                                       frameon=True, facecolor='white', edgecolor='black', title_fontproperties={'weight':'bold'})
            if main_legend:
                ax.add_artist(main_legend)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)

def load_transport_network(csv_filepath):
    df = pd.read_csv(csv_filepath)
    G = nx.Graph()
    
    for _, row in df.iterrows():
        node_id = row['code']
        node_attrs = row.drop('code').to_dict()
        G.add_node(node_id, **node_attrs)
        
        next_node = row.get('next_node')
        if pd.notna(next_node):
            edge_attrs = {}
            for col in ['edge_type', 'edge_color', 'curve_side', 'curve_smoothness', 
                        'curve_bulge', 'linewidth', 'distance_km', 'distance_mile', 'line']:
                if col in row and pd.notna(row[col]):
                    edge_attrs[col] = row[col]
            G.add_edge(node_id, next_node, **edge_attrs)
    
    # Extract nodes data for drawing 
    nodes_data = {}
    coord_labels_temp = {}
    for node_id, data in G.nodes(data=True):
        x = data.get('x_visual')
        y = data.get('y_visual')
        if x is None or y is None or pd.isna(x) or pd.isna(y):
            continue
        coord = (x, y)
        color = data.get('color', 'white') 
        if pd.isna(color): 
            color = 'white'
        station_name = data.get('station_name', node_id)
        if pd.isna(station_name):
            station_name = str(node_id)
        else:
            station_name = str(station_name)
        
        if coord not in nodes_data:
            x_text = data.get('x-text', 0)
            y_text = data.get('y-text', 0)
            ha = data.get('ha', 'center')
            va = data.get('va', 'center')
            nodes_data[coord] = {
                'colors': [],
                'offset': (0 if pd.isna(x_text) else x_text, 
                           0 if pd.isna(y_text) else y_text),
                'ha': 'center' if pd.isna(ha) else ha,
                'va': 'center' if pd.isna(va) else va
            }
            coord_labels_temp[coord] = []
        
        if color not in nodes_data[coord]['colors']:
            nodes_data[coord]['colors'].append(color)
        if station_name not in coord_labels_temp[coord]:
            coord_labels_temp[coord].append(station_name)
    
    for coord in nodes_data:
        nodes_data[coord]['label'] = " / ".join(coord_labels_temp[coord])

    # Extract edges
    edges_data = []
    for u, v, edge_attrs in G.edges(data=True):        
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        x1, y1 = u_data.get('x_visual'), u_data.get('y_visual')
        x2, y2 = v_data.get('x_visual'), v_data.get('y_visual')
        
        if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
            continue

        edge_type = edge_attrs.get('edge_type', 'straight')
        edge_color = edge_attrs.get('edge_color', 'gray')
        curve_side = edge_attrs.get('curve_side', 'right')
        curve_smoothness = edge_attrs.get('curve_smoothness', 100)
        curve_bulge = edge_attrs.get('curve_bulge', 0.2)
        linewidth = edge_attrs.get('linewidth', 3.0)

        dist_km = edge_attrs.get('distance_km', 0)
        distance_km_label = f"{dist_km:2.2f}" if not pd.isna(dist_km) else ''
        dist_mile = edge_attrs.get('distance_mile', 0)
        distance_mile_label = f"{dist_mile:2.2f}" if not pd.isna(dist_mile) else ''
        line = edge_attrs.get('line', '')

        edges_data.append({
            'p1': (x1, y1),
            'p2': (x2, y2),
            'type': edge_type,
            'color': edge_color,
            'linewidth': linewidth,
            'curve_side': curve_side,
            'curve_smoothness': curve_smoothness,
            'curve_bulge': curve_bulge,
            'distance_label': distance_km_label,
            'distance_mile_label': distance_mile_label,
            'line': line,
            'distance_km': float(dist_km) if not pd.isna(dist_km) else 0.0,
            'distance_mile': float(dist_mile) if not pd.isna(dist_mile) else 0.0
        })
        
    return G, nodes_data, edges_data

def calculate_task2_stats(edges_data):
    total_km = 0.0
    total_mile = 0.0
    edge_count = len(edges_data)

    for edge in edges_data:
        total_km += edge.get('distance_km', 0.0)
        total_mile += edge.get('distance_mile', 0.0)

    avg_km = total_km / edge_count if edge_count > 0 else 0.0
    avg_mile = total_mile / edge_count if edge_count > 0 else 0.0

    return total_km, total_mile, avg_km, avg_mile

def draw_nodes(ax, nodes_data, node_size=300):
    for (x, y), data in nodes_data.items():
        colors = data['colors']
        if len(colors) == 1:
            ax.scatter(x, y, s=node_size, facecolors=colors[0], edgecolors='black', linewidths=1.5, zorder=5)
        elif len(colors) >= 2:
            marker_left = MarkerStyle('o', fillstyle='left')
            ax.scatter(x, y, s=node_size, marker=marker_left, facecolors=colors[0], edgecolors='black', linewidths=1.5, zorder=5)
            marker_right = MarkerStyle('o', fillstyle='right')
            ax.scatter(x, y, s=node_size, marker=marker_right, facecolors=colors[1], edgecolors='black', linewidths=1.5, zorder=5)

        ax.annotate(data['label'], (x, y), 
                    textcoords="offset points", 
                    xytext=data['offset'], 
                    ha=data['ha'], 
                    va=data['va'], 
                    fontsize=9, 
                    fontweight='bold', 
                    zorder=10)

def draw_edges(ax, edges_data, choose_unit='1'):
    text_bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=1)

    for edge in edges_data:
        x1, y1 = edge['p1']
        x2, y2 = edge['p2']
        edge_type = edge.get('type', 'straight')
        color = edge.get('color', 'gray')
        linewidth = edge.get('linewidth', 3.0)
        distance_label = edge.get('distance_label', '') if choose_unit == '1' else edge.get('distance_mile_label', '')
        distance_label += " km" if choose_unit == '1' else " mi"
        
        if edge_type == 'straight':
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, zorder=1)
            if distance_label:
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                ax.text(mx, my, distance_label, fontsize=8, color='black', 
                        ha='center', va='center', bbox=text_bbox, zorder=2)
            
        elif edge_type == 'curve':
            curve_side = edge.get('curve_side', 'right')
            curve_smoothness = edge.get('curve_smoothness', 100)
            curve_bulge = edge.get('curve_bulge', 0.2)

            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if length == 0: continue

            nx_ = dx / length
            ny_ = dy / length

            if curve_side == 'left':
                px, py = -ny_, nx_
            else:  
                px, py = ny_, -nx_

            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2

            cx = mx + px * curve_bulge * length
            cy = my + py * curve_bulge * length

            t = np.linspace(0, 1, curve_smoothness)
            bx = (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
            by = (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2

            ax.plot(bx, by, color=color, linewidth=linewidth, zorder=1)

            if distance_label:
                t_mid = 0.5
                mx_curve = (1 - t_mid)**2 * x1 + 2 * (1 - t_mid) * t_mid * cx + t_mid**2 * x2
                my_curve = (1 - t_mid)**2 * y1 + 2 * (1 - t_mid) * t_mid * cy + t_mid**2 * y2
                ax.text(mx_curve, my_curve, distance_label, fontsize=8, color='black', 
                        ha='center', va='center', bbox=text_bbox, zorder=2)

def draw_legend(ax, edges_data, choose_unit='1', 
                total_km=None, total_mile=None, 
                avg_km=None, avg_mile=None):
    legend_elements = []

    unique_colors = {edge.get('color', 'gray'): edge.get('line', 'Line') 
                     for edge in edges_data}
    for color, line in unique_colors.items():
        if color != 'gray':
            legend_elements.append(Line2D([0], [0], color=color, lw=3, label=f"{line}"))

    msize = 11
    normal_station = Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='grey', markeredgecolor='black',
                            markersize=msize, markeredgewidth=1.5,
                            label='Normal Station')
    interchange_station = Line2D([0], [0], marker='o', color='w',
                                 fillstyle='left',
                                 markerfacecolor='grey',
                                 markerfacecoloralt='white',
                                 markeredgecolor='black',
                                 markersize=msize, markeredgewidth=1.5,
                                 label='Interchange Station')
    
    legend_elements.extend([normal_station, interchange_station])

    unit_label = "Distance: Kilometers (km)" if choose_unit == '1' else "Distance: Miles (mi)"
    legend_elements.append(Line2D([0], [0], marker='none', linestyle='none', label=unit_label))

    if total_km is not None:
        legend_elements.append(Line2D([0], [0], marker='none', linestyle='none', label=f"----------------------------------------------"))
        legend_elements.append(Line2D([0], [0], marker='none', linestyle='none', label=f"Total length: {total_km:.2f} km | {total_mile:.2f} mi"))
        legend_elements.append(Line2D([0], [0], marker='none', linestyle='none', label=f"Avg distance: {avg_km:.2f} km | {avg_mile:.2f} mi"))

    ax.legend(handles=legend_elements, loc='lower right', title="Key", 
              frameon=True, facecolor='white', edgecolor='black', 
              title_fontproperties={'weight':'bold'}, fontsize=9)

# ====================== MAIN PROGRAM ======================
G, nodes_ready_to_draw, edges_ready_to_draw = load_transport_network('data.csv')

print("Hello! This is a graph visualization of the transit system. You can choose to display distances in either kilometers or miles.")
print("1. Kilometers (km)")
print("2. Miles (mile)")
print("0. Exit")
choose_unit = input("Choose distance unit to display: ").strip()

while choose_unit not in ['1', '2', '0']:
    print("Invalid choice. Please enter 1 for kilometers, 2 for miles, or 0 to exit.")
    choose_unit = input("Choose distance unit to display: ").strip()


if choose_unit == '0':
    print("Exiting the program. Goodbye!")
    exit()

total_km, total_mile, avg_km, avg_mile = calculate_task2_stats(edges_ready_to_draw)

fig, ax = plt.subplots(figsize=(20, 17))
bg_color = '#fcfcfc'
fig.patch.set_facecolor(bg_color)

ax.set_aspect('equal', adjustable='datalim')
ax.axis('off')

draw_edges(ax, edges_ready_to_draw, choose_unit=choose_unit)
draw_nodes(ax, nodes_ready_to_draw, node_size=300)
draw_legend(ax, edges_ready_to_draw, choose_unit=choose_unit,
            total_km=total_km, total_mile=total_mile,
            avg_km=avg_km, avg_mile=avg_mile)

add_interactivity(fig, ax, G, nodes_ready_to_draw, node_size=300)

plt.title('Singapore Public Transport Network', fontweight='bold', pad=15)
plt.axis('equal')
plt.savefig('graph.png', bbox_inches='tight', pad_inches=0.1)
plt.show()