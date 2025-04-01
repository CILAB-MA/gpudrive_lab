import pygame
import pygame.gfxdraw
import numpy as np
import math
import gpudrive
import matplotlib.cm as cm
from shapely.geometry import Point, Polygon

from pygpudrive.env.config import MadronaOption, PygameOption, RenderMode
from pygpudrive.env.constants import MAX_REL_AGENT_POS

# AGENT COLORS
PINK = (255, 105, 180)
GREEN = (113, 228, 0)
BLUE = (0, 127, 255)
DODGER_BLUE = (30, 144, 255)
RED_ORANGE = (255, 69, 0)
WHITE = (255, 255, 255)
CHARCOAL = (22, 28, 32)

STATIC_AGENT_ID = 2

SHORT_MIN = np.iinfo(np.int16).min  # -32768
SHORT_MAX = np.iinfo(np.int16).max  # 32767



class PyGameVisualizer:
    WINDOW_W, WINDOW_H = 1920, 1080
    PADDING_PCT = 0.0
    COLOR_LIST = [
        BLUE,
        PINK,
        GREEN,
        DODGER_BLUE,
        RED_ORANGE,
    ]

    def __init__(self, sim, render_config, goal_radius):
        self.sim = sim
        self.render_config = render_config
        self.WINDOW_W, self.WINDOW_H = render_config.resolution
        self.goal_radius = goal_radius
        
        if self.render_config.color_scheme == "light":
            self.BACKGROUND_COLOR = WHITE
            self.vehicle_idx_color = CHARCOAL
        else:
            self.BACKGROUND_COLOR = CHARCOAL
            self.vehicle_idx_color = WHITE
        
        # ROAD MAP COLORS
        self.color_dict = {
            float(gpudrive.EntityType.RoadEdge): (68, 193, 123) if self.render_config.color_scheme == "dark" else (47,79,79),
            float(gpudrive.EntityType.RoadLine): (255, 245, 99),  # Yellow
            float(gpudrive.EntityType.RoadLane): (225, 225, 225),  # Grey
            float(gpudrive.EntityType.SpeedBump): (138, 43, 226),  # Purple
            float(gpudrive.EntityType.CrossWalk): (255, 255, 255),  # White
            float(gpudrive.EntityType.StopSign): (213, 20, 20),  # Dark red
            float(gpudrive.EntityType.Vehicle): (0, 255, 0),  # Green
            float(gpudrive.EntityType.Pedestrian): (0, 255, 0),  # Green
            float(gpudrive.EntityType.Cyclist): (0, 0, 255),  # Blue
        }

        self.num_agents = self.sim.shape_tensor().to_torch().cpu().numpy()
        self.num_worlds = self.sim.shape_tensor().to_torch().shape[0]

        self.padding_x = self.PADDING_PCT * self.WINDOW_W
        self.padding_y = self.PADDING_PCT * self.WINDOW_H

        self.zoom_scales_x = np.array([1.0] * self.num_worlds)
        self.zoom_scales_y = np.array([1.0] * self.num_worlds)
        self.window_centers = np.array([[0, 0]] * self.num_worlds)

        if self.render_config.render_mode in {
            RenderMode.PYGAME_ABSOLUTE,
            RenderMode.PYGAME_EGOCENTRIC,
            RenderMode.PYGAME_LIDAR,
        }:
            pygame.init()
            pygame.font.init()
            self.screen = None
            self.clock = None
            if (
                self.screen is None
                and self.render_config.view_option == PygameOption.HUMAN
            ):
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.WINDOW_W, self.WINDOW_H)
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.surf = pygame.Surface((self.WINDOW_W, self.WINDOW_H))
            self.compute_window_settings()
        if self.render_config.render_mode == RenderMode.PYGAME_ABSOLUTE:
            if self.render_config.draw_expert_footprint:
                self.footprints = np.zeros((gpudrive.episodeLen, self.num_worlds, gpudrive.kMaxAgentCount, 2))
            else:
                self.footprints = None
            if self.render_config.draw_other_aux:
                self.ego_aux = np.zeros((self.num_worlds, gpudrive.episodeLen, 3))
                self.other_aux = np.zeros((self.num_worlds, gpudrive.episodeLen, 128, 3))
            else:
                self.ego_aux = None
                self.other_aux = None

    @staticmethod
    def get_all_endpoints(map_info):
        centers = map_info[:, :2]
        lengths = map_info[:, 2]
        yaws = map_info[:, 5]

        offsets = np.column_stack(
            (lengths * np.cos(yaws), lengths * np.sin(yaws))
        )
        starts = centers - offsets
        ends = centers + offsets
        return starts, ends

    def draw_line(self, surf, start, end, color, thickness=1, fill_shape=True):
        c1 = (start[0] + end[0]) / 2
        c2 = (start[1] + end[1]) / 2
        center_L1 = (c1, c2)
        length = math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        angle = math.atan2(start[1] - end[1], start[0] - end[0])

        UL = (
            center_L1[0]
            + (length / 2.0) * np.cos(angle)
            - (thickness / 2.0) * np.sin(angle),
            center_L1[1]
            + (thickness / 2.0) * np.cos(angle)
            + (length / 2.0) * np.sin(angle),
        )
        UR = (
            center_L1[0]
            - (length / 2.0) * np.cos(angle)
            - (thickness / 2.0) * np.sin(angle),
            center_L1[1]
            + (thickness / 2.0) * np.cos(angle)
            - (length / 2.0) * np.sin(angle),
        )
        BL = (
            center_L1[0]
            + (length / 2.0) * np.cos(angle)
            + (thickness / 2.0) * np.sin(angle),
            center_L1[1]
            - (thickness / 2.0) * np.cos(angle)
            + (length / 2.0) * np.sin(angle),
        )
        BR = (
            center_L1[0]
            - (length / 2.0) * np.cos(angle)
            + (thickness / 2.0) * np.sin(angle),
            center_L1[1]
            - (thickness / 2.0) * np.cos(angle)
            - (length / 2.0) * np.sin(angle),
        )

        if fill_shape:
            pygame.gfxdraw.aapolygon(surf, (UL, UR, BR, BL), color)
            pygame.gfxdraw.filled_polygon(surf, (UL, UR, BR, BL), color)
        else:
            pygame.gfxdraw.aapolygon(surf, (UL, UR, BR, BL), color)

    def draw_circle(self, surf, center, radius, color, thickness=3):
        for i in range(thickness):
            try:
                pygame.gfxdraw.aacircle(
                    surf,
                    int(center[0]),
                    int(center[1]),
                    int(radius) + i,
                    color,
                )
            except:
                continue

    def compute_window_settings(self, map_infos=None):
        if map_infos is None:
            map_infos = (
                self.sim.map_observation_tensor().to_torch().cpu().numpy()
            )
        assert map_infos.shape[0] <= self.num_worlds
        for i in range(map_infos.shape[0]):
            map_info = map_infos[i]
            map_info = map_info[
                map_info[:, -1] != float(gpudrive.EntityType.Padding)
            ]
            roads = map_info[
                map_info[:, -1] <= float(gpudrive.EntityType.RoadLane)
            ]
            endpoints = PyGameVisualizer.get_all_endpoints(roads)

            all_endpoints = np.concatenate(endpoints, axis=0)

            # Adjust window dimensions by subtracting padding
            adjusted_window_width = self.WINDOW_W - self.padding_x
            adjusted_window_height = self.WINDOW_H - self.padding_y

            self.zoom_scales_x[i] = adjusted_window_width / (
                all_endpoints[:, 0].max() - all_endpoints[:, 0].min()
            )
            self.zoom_scales_y[i] = adjusted_window_height / (
                all_endpoints[:, 1].max() - all_endpoints[:, 1].min()
            )

            self.window_centers[i] = np.array(
                [
                    (all_endpoints[:, 0].max() + all_endpoints[:, 0].min())
                    / 2,
                    (all_endpoints[:, 1].max() + all_endpoints[:, 1].min())
                    / 2,
                ]
            )

    def scale_coords(self, coords, world_render_idx):
        """Scale the coordinates to fit within the pygame surface window and center them.
        Args:
            coords: x, y coordinates
        """
        x, y = coords
        x_scaled = (
            (x - self.window_centers[world_render_idx][0])
            * self.zoom_scales_x[world_render_idx]
            + self.WINDOW_W / 2
            - self.padding_x / 2
        )
        y_scaled = (
            (y - self.window_centers[world_render_idx][1])
            * self.zoom_scales_y[world_render_idx]
            + self.WINDOW_H / 2
            - self.padding_y / 2
        )

        return (x_scaled, y_scaled)

    @staticmethod
    def compute_agent_corners(center, width, height, rotation):
        """Draw a rectangle, centered at x, y.

        Arguments:
        x (int/float):
            The x coordinate of the center of the shape.
        y (int/float):
            The y coordinate of the center of the shape.
        width (int/float):
            The width of the rectangle.
        height (int/float):
            The height of the rectangle.
        """
        x, y = center

        points = []

        # The distance from the center of the rectangle to
        # one of the corners is the same for each corner.
        radius = math.sqrt((height / 2) ** 2 + (width / 2) ** 2)

        # Get the angle to one of the corners with respect
        # to the x-axis.
        angle = math.atan2(height / 2, width / 2)

        # Adjust angles for Pygame, where 0 angle is to the right
        # and rotations are clockwise
        angles = [
            angle - math.pi / 2 + rotation,
            math.pi - angle - math.pi / 2 + rotation,
            math.pi + angle - math.pi / 2 + rotation,
            -angle - math.pi / 2 + rotation,
        ]

        # Calculate the coordinates of each corner for Pygame
        for angle in angles:
            x_offset = radius * math.cos(angle)
            y_offset = radius * math.sin(angle)  # Invert y-coordinate
            points.append((x + x_offset, y + y_offset))

        return points

    @staticmethod
    def get_endpoints(center, map_obj):
        center_pos = center
        length = map_obj[2]  # Already half the length
        yaw = map_obj[5]

        start = center_pos - np.array(
            [length * np.cos(yaw), length * np.sin(yaw)]
        )
        end = center_pos + np.array(
            [length * np.cos(yaw), length * np.sin(yaw)]
        )
        return start, end

    def draw_map(self, surf, map_info, world_render_idx=0):
        """Draw static map elements."""
        for idx, map_obj in enumerate(map_info):

            if map_obj[-1] == float(gpudrive.EntityType.Padding) or map_obj[-1] == float(gpudrive.EntityType._None):
                continue

            elif map_obj[-1] <= float(gpudrive.EntityType.RoadLane):
                start, end = PyGameVisualizer.get_endpoints(
                    map_obj[:2], map_obj
                )
                start = self.scale_coords(start, world_render_idx)
                end = self.scale_coords(end, world_render_idx)

                # DRAW ROAD EDGE
                if map_obj[-1] == float(gpudrive.EntityType.RoadEdge):
                    self.draw_line(
                        surf,
                        start,
                        end,
                        self.color_dict[map_obj[-1]],
                        thickness=self.render_config.line_thickness,
                    )

                # DRAW ROAD LINES/LANES
                else:
                    self.draw_line(
                        surf,
                        start,
                        end,
                        self.color_dict[map_obj[-1]],
                        thickness=self.render_config.line_thickness,
                    )

            # DRAW STOP SIGNS
            elif map_obj[-1] <= float(gpudrive.EntityType.StopSign):

                center, width, height, rotation = (
                    map_obj[:2],
                    map_obj[3],
                    map_obj[2],
                    map_obj[5],
                )
                if map_obj[-1] == float(gpudrive.EntityType.StopSign):

                    width *= self.zoom_scales_x[world_render_idx]
                    height *= self.zoom_scales_y[world_render_idx]

                box_corners = PyGameVisualizer.compute_agent_corners(
                    center, width, height, rotation
                )
                for i, box_corner in enumerate(box_corners):
                    box_corners[i] = self.scale_coords(
                        box_corner, world_render_idx
                    )
                if map_obj[-1] == float(gpudrive.EntityType.SpeedBump):
                    pygame.gfxdraw.aapolygon(
                        surf, box_corners, self.color_dict[map_obj[-1]]
                    )
                else:
                    pygame.gfxdraw.aapolygon(
                        surf, box_corners, self.color_dict[map_obj[-1]]
                    )
                    pygame.gfxdraw.filled_polygon(
                        surf, box_corners, self.color_dict[map_obj[-1]]
                    )

    def init_map(self):
        """Initialize the static map elements."""

        if self.render_config.render_mode == RenderMode.PYGAME_EGOCENTRIC:
            return

        self.map_surf.fill(self.BACKGROUND_COLOR)
        self.map_surfs = []
        for i in range(self.num_worlds):
            map_surf = self.surf.copy()
            map_info = (
                self.sim.map_observation_tensor().to_torch()[i].cpu().numpy()
            )
            self.draw_map(map_surf, map_info, i)
            self.map_surfs.append(map_surf)

    def getRender(
        self, world_render_idx=0, time_step=0, color_objects_by_actor=None, **kwargs
    ):
        if self.render_config.render_mode in {
            RenderMode.PYGAME_ABSOLUTE,
            RenderMode.PYGAME_EGOCENTRIC,
            RenderMode.PYGAME_LIDAR,
        }:
            cont_agent_mask = kwargs.get("cont_agent_mask", None)
            return self.draw(
                cont_agent_mask, world_render_idx, time_step, color_objects_by_actor
            )
        elif self.render_config.render_mode == RenderMode.MADRONA_RGB:
            if self.render_config.view_option == MadronaOption.TOP_DOWN:
                raise NotImplementedError
            return self.sim.rgb_tensor().to_torch()
        elif self.render_config.render_mode == RenderMode.MADRONA_DEPTH:
            if self.render_config.view_option == MadronaOption.TOP_DOWN:
                raise NotImplementedError
            return self.sim.depth_tensor().to_torch()

    def plotLidar(self, surf, lidar_data, world_render_idx):
        numLidarSamples = lidar_data.shape[0]

        lidar_entity_types = lidar_data[:, 1]
        lidar_pos = lidar_data[:, 2:4]

        for i in range(numLidarSamples):
            if(lidar_entity_types[i] == float(gpudrive.EntityType._None)):
                continue
            coords = lidar_pos[i]
            scaled_coords = self.scale_coords(coords, world_render_idx)
           
            pygame.gfxdraw.aacircle(
                surf,
                int(scaled_coords[0]),
                int(scaled_coords[1]),
                2,
                self.color_dict[lidar_entity_types[i]]
            )

            pygame.gfxdraw.filled_circle(
                surf,
                int(scaled_coords[0]),
                int(scaled_coords[1]),
                2,
                self.color_dict[lidar_entity_types[i]]
            )

    def draw(
        self, cont_agent_mask, world_render_idx=0, time_step=0, color_objects_by_actor=None
    ):
        """Render the environment."""

        if self.render_config.render_mode == RenderMode.PYGAME_EGOCENTRIC:
            render_rgbs = []
            num_agents = self.num_agents[world_render_idx][0]
            # Loop through each agent to render their egocentric view
            for agent_idx in range(num_agents):
                info_tensor = self.sim.info_tensor().to_torch()[
                    world_render_idx
                ]
                if info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType.Padding
                ) or info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType._None
                ):
                    continue

                self.surf.fill(self.BACKGROUND_COLOR)
                agent_map_info = (
                    self.sim.agent_roadmap_tensor()
                    .to_torch()[world_render_idx, agent_idx, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                )
                agent_map_info = agent_map_info[
                    (agent_map_info[:, -1] != 0.0)
                    & (agent_map_info[:, -1] != 10.0)
                ]

                agent_info = (
                    self.sim.self_observation_tensor()
                    .to_torch()[world_render_idx, agent_idx, :]
                    .cpu()
                    .detach()
                    .numpy()
                )

                partner_agent_info = (
                    self.sim.partner_observations_tensor()
                    .to_torch()[world_render_idx, agent_idx, :, :]
                    .cpu()
                    .detach()
                    .numpy()
                )
                partner_agent_info = partner_agent_info[
                    partner_agent_info[:, -1] == 7.0
                ]

                goal_pos = agent_info[3:5]  # x, y
                agent_size = agent_info[1:3]  # length, width

                # Create a temporary surface for the egocentric view
                temp_surf = pygame.Surface(
                    (self.surf.get_width(), self.surf.get_height())
                )
                temp_surf.fill(self.BACKGROUND_COLOR)

                self.draw_map(temp_surf, agent_map_info)
                # Transform the map surface to the agent's egocentric view
                agent_corners = PyGameVisualizer.compute_agent_corners(
                    (0, 0), agent_size[1], agent_size[0], 0
                )
                agent_corners = [
                    self.scale_coords(corner, world_render_idx)
                    for corner in agent_corners
                ]
                current_goal_scaled = self.scale_coords(
                    goal_pos, world_render_idx
                )

                pygame.gfxdraw.aapolygon(
                    temp_surf, agent_corners, self.COLOR_LIST[0]
                )
                pygame.gfxdraw.filled_polygon(
                    temp_surf, agent_corners, self.COLOR_LIST[0]
                )

                self.draw_circle(
                    temp_surf,
                    current_goal_scaled,
                    self.goal_radius * self.zoom_scales_x[world_render_idx],
                    self.COLOR_LIST[0],
                )

                for agent in partner_agent_info:
                    agent_pos = agent[1:3]
                    agent_rot = agent[3]
                    agent_size = agent[4:6]
                    agent_type = agent[-1]

                    agent_corners = PyGameVisualizer.compute_agent_corners(
                        agent_pos,
                        agent_size[1],
                        agent_size[0],
                        agent_rot,
                    )

                    agent_corners = [
                        self.scale_coords(corner, world_render_idx)
                        for corner in agent_corners
                    ]

                    pygame.gfxdraw.aapolygon(
                        temp_surf, agent_corners, self.COLOR_LIST[1]
                    )
                    pygame.gfxdraw.filled_polygon(
                        temp_surf,
                        agent_corners,
                        self.COLOR_LIST[1],
                    )

                # blit temp surf on self.surf
                self.surf.blit(temp_surf, (0, 0))
                # Capture the RGB array for the agent's view
                render_rgbs.append(
                    PyGameVisualizer._create_image_array(self.surf)
                )

            return render_rgbs
        elif self.render_config.render_mode == RenderMode.PYGAME_ABSOLUTE:
            self.surf.fill(self.BACKGROUND_COLOR)
            map_info = (
                self.sim.map_observation_tensor()
                .to_torch()[world_render_idx]
                .cpu()
                .numpy()
            )
            self.draw_map(self.surf, map_info, world_render_idx)
            # Get agent info
            agent_info = (
                self.sim.absolute_self_observation_tensor()
                .to_torch()[world_render_idx, :, :]
                .cpu()
                .detach()
                .numpy()
            )

            # Get the agent goal positions and current positions
            agent_pos = agent_info[:, :2]  # x, y
            goal_pos = agent_info[:, 8:10]  # x, y
            agent_rot = agent_info[:, 7]  # heading
            agent_sizes = agent_info[:, 10:12]  # length, width
            agent_response_types = (  # 0: Valid (can be controlled), 2: Invalid (static vehicles)
                self.sim.response_type_tensor()
                .to_torch()[world_render_idx, :, :]
                .cpu()
                .detach()
                .numpy()
            )

            num_agents = self.num_agents[world_render_idx][0]
            if color_objects_by_actor is not None:
                categories = list(color_objects_by_actor.keys())

            valid_agent_indices = list(
                range((agent_response_types == 0).sum())
            )

            # Draw the agent positions
            partner_color = []
            for agent_idx in range(num_agents):

                if color_objects_by_actor is not None:
                    for i in range(len(categories)):
                        if agent_idx in color_objects_by_actor[categories[i]]:
                            mod_idx = i
                            break
                    else:
                        mod_idx = 3
                else:
                    mod_idx = agent_idx % len(self.COLOR_LIST)

                color = self.COLOR_LIST[mod_idx] if not self.render_config.draw_ego_attention else (128, 128, 128)

                info_tensor = self.sim.info_tensor().to_torch()[
                    world_render_idx
                ]
                if info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType.Padding
                ) or info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType._None
                ):
                    continue

                agent_corners = PyGameVisualizer.compute_agent_corners(
                    agent_pos[agent_idx],
                    agent_sizes[agent_idx, 1],
                    agent_sizes[agent_idx, 0],
                    agent_rot[agent_idx],
                )

                for i, agent_corner in enumerate(agent_corners):
                    agent_corners[i] = self.scale_coords(
                        agent_corner, world_render_idx
                    )

                current_goal_scaled = self.scale_coords(
                    goal_pos[agent_idx], world_render_idx
                )

                # Agent is static
                if agent_response_types[agent_idx] == STATIC_AGENT_ID:
                    color = (128, 128, 128)
                else:
                    partner_color.append(color)

                # Draw the expert footprint
                if agent_response_types[agent_idx] != STATIC_AGENT_ID and self.footprints is not None:
                    if (self.render_config.draw_only_ego_footprint and cont_agent_mask[world_render_idx][agent_idx]) or not self.render_config.draw_only_ego_footprint:
                        for footprint in self.footprints:
                            x = footprint[world_render_idx, agent_idx, 0]
                            y = footprint[world_render_idx, agent_idx, 1]
                            if SHORT_MIN <= x <= SHORT_MAX and SHORT_MIN <= y <= SHORT_MAX:
                                pygame.gfxdraw.filled_circle(
                                    self.surf, int(x), int(y), 2, color
                                )
                        
                pygame.gfxdraw.aapolygon(self.surf, agent_corners, color)
                pygame.gfxdraw.filled_polygon(self.surf, agent_corners, color)

                # Draw object indices for the controllable agents
                if (
                    self.render_config.draw_obj_idx
                    and agent_response_types[agent_idx] != STATIC_AGENT_ID
                ):
                    scaled_font_size = (
                        self.render_config.obj_idx_font_size
                        * int(self.zoom_scales_x[world_render_idx])
                    )
                    font = pygame.font.Font(None, scaled_font_size)
                    text = font.render(
                        str(valid_agent_indices.pop(0)), True, self.vehicle_idx_color
                    )
                    text_rect = text.get_rect(
                        center=(
                            agent_corners[0][0]
                            - 2 * int(self.zoom_scales_y[0]),
                            agent_corners[0][1],
                        )
                    )
                    self.surf.blit(text, text_rect)

                if agent_response_types[agent_idx] != STATIC_AGENT_ID:
                    self.draw_circle(
                        self.surf,
                        current_goal_scaled,
                        self.goal_radius
                        * self.zoom_scales_x[world_render_idx],
                        color,
                    )

            if self.render_config.draw_other_aux:
                partner_color = partner_color[1:]
                self.draw_other_auxiliary(world_render_idx, time_step, agent_response_types, partner_color)
            
            if self.render_config.draw_ego_attention:
                self.attn_surfs = [self.surf.copy() for _ in range(self.ego_attn_score.shape[1])]
                self.draw_attention(agent_info, world_render_idx)
            
            if self.render_config.view_option == PygameOption.HUMAN:
                pygame.event.pump()
                self.clock.tick(self.metadata["render_fps"])
                assert self.screen is not None
                self.screen.fill(0)
                self.screen.blit(self.surf, (0, 0))
                pygame.display.flip()
            elif self.render_config.view_option == PygameOption.RGB:
                if self.render_config.draw_ego_attention:
                    return PyGameVisualizer._create_attn_image_array(self.attn_surfs)
                else:    
                    return PyGameVisualizer._create_image_array(self.surf)
            else:
                return self.isopen
        elif self.render_config.render_mode == RenderMode.PYGAME_LIDAR:
            render_rgbs = []
            num_agents = self.num_agents[world_render_idx][0]

            # Loop through each agent to render their egocentric view
            for agent_idx in range(num_agents):
                info_tensor = self.sim.info_tensor().to_torch()[world_render_idx]
                if info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType.Padding
                ) or info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType._None
                ):
                    continue

                self.surf.fill(self.BACKGROUND_COLOR)
                temp_surf = pygame.Surface(
                    (self.surf.get_width(), self.surf.get_height())
                )
                temp_surf.fill(self.BACKGROUND_COLOR)

                agent_info = (
                    self.sim.self_observation_tensor()
                    .to_torch()[world_render_idx, agent_idx, :]
                    .cpu()
                    .detach()
                    .numpy()
                )

                lidar_data = (
                    self.sim.lidar_tensor()
                    .to_torch()[world_render_idx, agent_idx, :, :, :] # shape is (num_worlds, num_agents, num_planes, num_samples, 2)
                    .cpu()
                    .detach()
                    .numpy()
                )
                
                for lidar_plane in lidar_data:
                    self.plotLidar(temp_surf, lidar_plane, world_render_idx)

                goal_pos = agent_info[3:5]  # x, y
                agent_size = agent_info[1:3]  # length, width

                agent_corners = PyGameVisualizer.compute_agent_corners(
                    (0, 0), agent_size[1], agent_size[0], 0
                )
                agent_corners = [
                    self.scale_coords(corner, world_render_idx)
                    for corner in agent_corners
                ]
                current_goal_scaled = self.scale_coords(
                    goal_pos, world_render_idx
                )

                pygame.gfxdraw.aapolygon(
                    temp_surf, agent_corners, self.COLOR_LIST[0]
                )
                pygame.gfxdraw.filled_polygon(
                    temp_surf, agent_corners, self.COLOR_LIST[0]
                )

                self.draw_circle(
                    temp_surf,
                    current_goal_scaled,
                    self.goal_radius * self.zoom_scales_x[world_render_idx],
                    self.COLOR_LIST[0],
                )

                # blit temp surf on self.surf
                self.surf.blit(temp_surf, (0, 0))
                # Capture the RGB array for the agent's view
                render_rgbs.append(
                    PyGameVisualizer._create_image_array(self.surf)
                )
            return render_rgbs

    @staticmethod
    def _create_image_array(surf):
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
        )
        
    @staticmethod
    def _create_attn_image_array(surfs):
        width, height = surfs[0].get_size()
        
        new_surf = pygame.Surface((width * 2, height * 2))
        
        new_surf.blit(surfs[0], (0, 0))
        new_surf.blit(surfs[1], (width, 0))
        new_surf.blit(surfs[2], (0, height))
        new_surf.blit(surfs[3], (width, height))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(new_surf)), axes=(1, 0, 2)
        )

    def destroy(self):
        pygame.display.quit()
        pygame.quit()

    def saveAux(self, world_render_idx=0, time_step=0):
        """save the aux for the agent at the given time step. (to use in drawing the expert trajectory)"""
        # Get expert position info
        agent_info = (
            self.sim.absolute_self_observation_tensor()
            .to_torch()[world_render_idx, :, :]
            .cpu()
            .detach()
            .numpy()
        )
        
        # Get the agent goal positions and current positions
        aux = np.concatenate((agent_info[:, :2], agent_info[:, 7:8]), axis=-1)  # x, y, heading
        
        agent_response_types = (  # 0: Valid (can be controlled), 2: Invalid (static vehicles)
            self.sim.response_type_tensor()
            .to_torch()[world_render_idx, :, :]
            .cpu()
            .detach()
            .numpy()
            .squeeze(axis=-1)
        )
        
        ego_idx = (agent_response_types == 0).nonzero()[0][0]

        info_tensor = self.sim.info_tensor().to_torch()[
            world_render_idx
        ]
        if info_tensor[ego_idx, -1] == float(
            gpudrive.EntityType.Padding
        ) or info_tensor[ego_idx, -1] == float(
            gpudrive.EntityType._None
        ):
            return
        
        self.other_aux[world_render_idx, time_step] = aux
        self.ego_aux[world_render_idx, time_step, :] = aux[ego_idx]
    
    def saveFootprint(self, world_render_idx=0, time_step=0, pos=None):
        """save the expert footprint for the agent at the given time step. (to use in drawing the expert trajectory)"""
        if pos is not None:
            agent_info = pos.cpu().detach().numpy()
        else:
            # Get expert position info
            agent_info = (
                self.sim.absolute_self_observation_tensor()
                .to_torch()[world_render_idx, :, :]
                .cpu()
                .detach()
                .numpy()
            )
            # Get the agent goal positions and current positions
            agent_pos = agent_info[:, :2]  # x, y
        agent_response_types = (  # 0: Valid (can be controlled), 2: Invalid (static vehicles)
            self.sim.response_type_tensor()
            .to_torch()[world_render_idx, :, :]
            .cpu()
            .detach()
            .numpy()
        )

        num_agents = self.num_agents[world_render_idx][0]

        # Draw the agent positions
        for agent_idx in range(num_agents):
            if agent_response_types[agent_idx] != STATIC_AGENT_ID:
                info_tensor = self.sim.info_tensor().to_torch()[
                    world_render_idx
                ]
                if info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType.Padding
                ) or info_tensor[agent_idx, -1] == float(
                    gpudrive.EntityType._None
                ):
                    continue
                
                x = self.scale_coords(agent_pos[agent_idx], world_render_idx)[0]
                y = self.scale_coords(agent_pos[agent_idx], world_render_idx)[1]
                
                self.footprints[time_step, world_render_idx, agent_idx] = [x, y]
                
                if self.render_config.draw_only_ego_footprint:
                    break

    def saveEgoAttnScore(self, ego_attn_score):
        """ego_attn_score: shape (num_worlds, num_multi_head, num_agents)"""
        setattr(self, "ego_attn_score", ego_attn_score)

    def draw_attention(self, agent_info, world_render_idx):
        # Get Attention score scaling
        ego_attn_score = self.ego_attn_score[world_render_idx]
        ego_attn_score = ego_attn_score.cpu().numpy()
        
        for head_idx in range(ego_attn_score.shape[0]):
            attn_score = ego_attn_score[head_idx]
            partner_ids = np.where(attn_score > 0)[0]
            mu = 1 / len(partner_ids) if len(partner_ids) != 0 else 0
            max_attn_score = attn_score.max()
            nonzero_scores = attn_score[attn_score > 0]
            if nonzero_scores.size > 0:
                min_attn_score = nonzero_scores.min()
            else:
                min_attn_score = attn_score.min()
            attn_score = (attn_score - min_attn_score) / (max_attn_score - min_attn_score + 1e-6)

            for partner_id in partner_ids:
                pos = agent_info[partner_id, :2]
                sizes = agent_info[partner_id, 10:12]
                rot = agent_info[partner_id, 7]

                agent_corners = PyGameVisualizer.compute_agent_corners(
                    pos,
                    sizes[1],
                    sizes[0],
                    rot,
                )

                for i, agent_corner in enumerate(agent_corners):
                        agent_corners[i] = self.scale_coords(
                            agent_corner, world_render_idx
                        )

                attention_strength = attn_score[partner_id]
                viridis_color = cm.viridis(attention_strength)
                color = tuple(int(c * 255) for c in viridis_color[:3])

                pygame.gfxdraw.aapolygon(self.attn_surfs[head_idx], agent_corners, color)
                pygame.gfxdraw.filled_polygon(self.attn_surfs[head_idx], agent_corners, color)

            self.draw_colorbar(min_attn_score, max_attn_score, mu, head_idx=head_idx)

    def draw_colorbar(self, min_val, max_val, mu, head_idx):
        width, height = 300, 30
        colorbar_surface = pygame.Surface((width, height))

        colorbar_array = np.linspace(0, 1, width).reshape(1, width)
        colorbar_image = cm.viridis(colorbar_array)[:, :, :3]
        colorbar_image = (colorbar_image * 255).astype(np.uint8)

        pygame.surfarray.blit_array(colorbar_surface, colorbar_image.swapaxes(0, 1))

        screen_width, screen_height = self.attn_surfs[head_idx].get_size()
        colorbar_position = (screen_width - width - 20 - 20, screen_height - height - 20 - 20)

        self.attn_surfs[head_idx].blit(colorbar_surface, colorbar_position)
        mu_position = int((mu - min_val) / (max_val - min_val + 1e-6) * width)
        mu_position = max(0, min(mu_position, width - 1))
        pygame.draw.line(
            self.attn_surfs[head_idx],
            (255, 0, 0),
            (colorbar_position[0] + mu_position, colorbar_position[1] - 2),
            (colorbar_position[0] + mu_position, colorbar_position[1] + height + 2),
            2 
        )

        font = pygame.font.SysFont(None, 20)
        small_font = pygame.font.SysFont(None, 16)
        pygame.draw.rect(self.attn_surfs[head_idx], (0, 0, 0), (*colorbar_position, width, height), 2)

        # TODO: Fix title_dict if auxiliary tasks are added
        title_dict = {
            0: "Speed",
            1: "Pos",
            2: "Heading",
            3: "Action"
        }
        
        title_text = font.render(f"{title_dict[head_idx]} Attention Weight", True, (0, 0, 0))
        title_x = colorbar_position[0] + width // 2 - title_text.get_width() // 2
        title_y = colorbar_position[1] - 25
        self.attn_surfs[head_idx].blit(title_text, (title_x, title_y))
        min_text = small_font.render(f"{float(min_val):.3f}", True, (0, 0, 0))
        max_text = small_font.render(f"{float(max_val):.3f}", True, (0, 0, 0))
        self.attn_surfs[head_idx].blit(min_text, (colorbar_position[0] - 40, colorbar_position[1] + height // 2 - min_text.get_height() // 2))
        self.attn_surfs[head_idx].blit(max_text, (colorbar_position[0] + width + 10, colorbar_position[1] + height // 2 - max_text.get_height() // 2))

    def saveAuxPred(self, aux_pred):
        setattr(self, "aux_pred", aux_pred)

    def draw_other_auxiliary(self, world_render_idx, time_step, agent_response_types, partner_color):
        for future_step, aux_pred in self.aux_pred.items():
            grid = self.get_ego_grid(world_render_idx, time_step)
            self.draw_other_future(world_render_idx, aux_pred, time_step, 2*future_step, agent_response_types, partner_color, grid)
    
    def get_ego_grid(self, world_render_idx, time_step):
        """Get the ego grid on the surface."""
        try:
            ego_pos = self.ego_aux[world_render_idx, time_step, :2]
            ego_rot = self.ego_aux[world_render_idx, time_step, 2]
        except IndexError:
            return # ego is not in the scene at future
        
        grid_corners = np.linspace(-0.03, 0.03, 9)
        grid_x, grid_y = np.meshgrid(grid_corners, grid_corners)
        grid_points = np.stack([grid_x, grid_y], axis=-1)
        grid_points = grid_points * MAX_REL_AGENT_POS
        
        ego_cos, ego_sin = np.cos(ego_rot), np.sin(ego_rot)
        
        # rotation transformation
        global_grid = []
        for i in range(grid_points.shape[0]):
            row = []
            for j in range(grid_points.shape[1]):
                x, y = grid_points[i, j]
                rotated_x = x * ego_cos - y * ego_sin
                rotated_y = x * ego_sin + y * ego_cos
                global_x = rotated_x + ego_pos[0]
                global_y = rotated_y + ego_pos[1]
                screen_point = self.scale_coords((global_x, global_y), world_render_idx)
                row.append(screen_point)
            global_grid.append(row)
        
        return global_grid
    
    def draw_other_future(self, world_render_idx, aux_pred, time_step, future_step, agent_response_types, partner_color, grid):        
        # 0. if the ego is done, return
        controlled_agent_id = (agent_response_types == 0).nonzero()[0]
        ego_id = controlled_agent_id[0]
        controlled_agent_id = controlled_agent_id[1:]
        done = self.sim.done_tensor().to_torch()[world_render_idx, :, 0].cpu().detach().numpy().astype(bool)
        if done[ego_id]:
            return
        
        @staticmethod
        def _recover_pos_from_discrete(discrete_pos):
            bins = np.linspace(-0.1, 0.1, 9)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            x_bins = (discrete_pos // 8).astype(int)
            y_bins = (discrete_pos % 8).astype(int)
            
            x = bin_centers[x_bins]
            y = bin_centers[y_bins]
            
            pos = np.stack([x, y], axis=-1)
            return pos
        
        @staticmethod
        def _recover_action_from_discrete(discrete_action):
            discrete_action = np.clip(discrete_action, 0, 11).astype('int')
            bin_centers = np.array([
                (-np.pi + -2.0*np.pi/3)/2,
                (-2.0*np.pi/3 + -np.pi/3)/2,
                (-np.pi/3 + -np.pi/6)/2,
                (-np.pi/6 + -np.pi/12)/2,
                (-np.pi/12 + -np.pi/36)/2,
                (-np.pi/36 + 0)/2,
                (0 + np.pi/36)/2,
                (np.pi/36 + np.pi/12)/2,
                (np.pi/12 + np.pi/6)/2,
                (np.pi/6 + np.pi/3)/2,
                (np.pi/3 + 2.0*np.pi/3)/2,
                (2.0*np.pi/3 + np.pi)/2,
            ])
            return bin_centers[discrete_action]
        
        @staticmethod
        def _draw_gt_pos(surf, partner_color, grid, controlled_agent_id, partner_ids, agent_pos, future_step):
            def blend_with_white(color, ratio):
                r, g, b = color
                r = int(r + (255 - r) * ratio)
                g = int(g + (255 - g) * ratio)
                b = int(b + (255 - b) * ratio)
                return (r, g, b)
            
            fade_ratio = {
                10: 0.0,
                20: 0.3,
                30: 0.6,
                40: 0.8
            }

            line_width = {
                10: 4,
                20: 3,
                30: 2,
                40: 1
            }

            grid = np.array(grid)
            
            for id, partner_id in enumerate(controlled_agent_id):
                if partner_id in partner_ids:
                    pos = agent_pos[partner_id]
                    pos = self.scale_coords(pos, world_render_idx)
                    point = Point(pos[0], pos[1])
                    for i in range(8):
                        for j in range(8):
                            c1 = grid[i, j]
                            c2 = grid[i, j + 1]
                            c3 = grid[i + 1, j + 1]
                            c4 = grid[i + 1, j]
                            
                            polygon = Polygon([c1, c2, c3, c4])
                            
                            if polygon.contains(point):
                                cx = (c1[0] + c2[0] + c3[0] + c4[0]) / 4
                                cy = (c1[1] + c2[1] + c3[1] + c4[1]) / 4
                                w = (np.linalg.norm(np.array(c1) - np.array(c2)) + np.linalg.norm(np.array(c4) - np.array(c3))) / 2
                                h = (np.linalg.norm(np.array(c1) - np.array(c4)) + np.linalg.norm(np.array(c2) - np.array(c3))) / 2
                                square = [
                                    (cx - 0.1 * w, cy - 0.1 * h),
                                    (cx + 0.1 * w, cy - 0.1 * h),
                                    (cx + 0.1 * w, cy + 0.1 * h),
                                    (cx - 0.1 * w, cy + 0.1 * h)
                                ]
                                faded_color = blend_with_white(partner_color[id], fade_ratio[future_step])
                                
                                pygame.draw.polygon(surf, faded_color, square, width=0)
                                break

        # 1. Get the global future positions of the other agents
        try:
            ego_pos = self.ego_aux[world_render_idx, time_step, :2]
            other_future_pos = self.other_aux[world_render_idx, time_step + future_step, :, :2]
            ego_rot = self.ego_aux[world_render_idx, time_step, 2]
        except:
            return # ego is not in the scene at future
        
        ego_cos, ego_sin = np.cos(ego_rot), np.sin(ego_rot)
        
        aux_pred_pos = aux_pred['pos'].cpu().detach().numpy()
        alive_partner_ids = np.where(np.logical_and(~done, agent_response_types[:, 0] == 0))[0]
        alive_partner_ids = alive_partner_ids[alive_partner_ids != ego_id]
        other_pos = _recover_pos_from_discrete(aux_pred_pos[world_render_idx])
        other_pos = other_pos * MAX_REL_AGENT_POS
        
        norm = np.linalg.norm(other_pos, axis=-1, keepdims=True)
        unit_vec = other_pos / norm
        other_cos, other_sin = unit_vec[:, 0], unit_vec[:, 1]
        
        other_global_cos = other_cos * ego_cos - other_sin * ego_sin
        other_global_sin = other_cos * ego_sin + other_sin * ego_cos
        
        other_global_x = np.linalg.norm(other_pos, axis=-1) * other_global_cos + ego_pos[0]
        other_global_y = np.linalg.norm(other_pos, axis=-1) * other_global_sin + ego_pos[1]
        other_global_pos = np.stack([other_global_x, other_global_y], axis=-1)
        
        # 2. Get the future rotation of the other agents
        aux_pred_rot = aux_pred['action'].cpu().detach().numpy()
        other_rot = _recover_action_from_discrete(aux_pred_rot[world_render_idx])
        other_rot = other_rot + ego_rot
        
        # 3. Draw the future ground truth positions of the other agents
        _draw_gt_pos(self.surf, partner_color, grid, controlled_agent_id, alive_partner_ids, other_future_pos, future_step)
        
        # 3. Draw the future rotations of the other agents on their global positions
        for i, partner_id in enumerate(controlled_agent_id):
            if partner_id in alive_partner_ids:
                color = partner_color[i]
                
                pos = other_global_pos[partner_id]
                pos = self.scale_coords(pos, world_render_idx)
                rot = other_rot[partner_id]

                # draw arrow body
                arrow_length = 20
                end_x = pos[0] + arrow_length * np.cos(rot)
                end_y = pos[1] + arrow_length * np.sin(rot)

                end_pos = (int(end_x), int(end_y))
                start_pos = (int(pos[0]), int(pos[1]))

                pygame.draw.line(self.surf, color, start_pos, end_pos, 4)

                # draw arrow head
                arrow_head_len = 6
                arrow_angle = np.pi / 6

                left_x = end_x - arrow_head_len * np.cos(rot - arrow_angle)
                left_y = end_y - arrow_head_len * np.sin(rot - arrow_angle)
                right_x = end_x - arrow_head_len * np.cos(rot + arrow_angle)
                right_y = end_y - arrow_head_len * np.sin(rot + arrow_angle)

                pygame.draw.line(self.surf, color, end_pos, (int(left_x), int(left_y)), 4)
                pygame.draw.line(self.surf, color, end_pos, (int(right_x), int(right_y)), 4)
                
                # draw future step number
                font = pygame.font.SysFont(None, 20)
                step_text = font.render(str(future_step), True, (0, 0, 0))
                text_rect = step_text.get_rect(center=end_pos)
                self.surf.blit(step_text, text_rect)
