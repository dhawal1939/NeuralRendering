from manim import *
import os
from manim.mobject.geometry import ArrowTip, ArrowTriangleFilledTip, ArrowTriangleTip

class Method(Scene):

    def construct(self):

        '''
        DR BOX
        '''
        DR = RoundedRectangle(color=BLUE, fill_opacity=1)
        dr_name = MathTex('Differentiable\\\\ Renderer').scale(0.5)

        '''
        Optimizing textures
        '''
        optimizing_texture = Square().next_to(DR, 8*LEFT)
        optimizing_texture_text = MathTex('Optimizing\ Texture').scale(.5).next_to(optimizing_texture, DOWN)

        diffuse_input_arrow = Arrow(optimizing_texture.get_edge_center(RIGHT), DR.get_edge_center(LEFT))
        diffuse_optimized_arrow = Arrow(DR.get_edge_center(LEFT), optimizing_texture.get_edge_center(RIGHT))


        '''
        Proxy Geometry
        '''
        proxy_geometry_arrow = Arrow(DR.get_edge_center(DOWN) - [1, 1, 0] , DR.get_edge_center(DOWN) - [1, 0, 0]).set_color(RED)
        proxy_geometery = MathTex('Proxy\ Geometry').next_to(proxy_geometry_arrow.get_end(), 2*DOWN).scale(.25)

        '''
        Camera Parameters
        '''
        camera_params_arrow = Arrow(DR.get_edge_center(DOWN) + [1, -1, 0], DR.get_edge_center(DOWN) + [1, 0, 0]).set_color(RED)
        camera_params = MathTex('Camera\ Parameters').next_to(camera_params_arrow.get_end(), 2*DOWN).scale(.25)


        '''
        Env Map
        '''
        env_map_arrow = Arrow(DR.get_edge_center(UP) + [0, 1, 0], DR.get_edge_center(UP)).set_color(RED)
        env_map = Rectangle().next_to(env_map_arrow.get_end(), .25*UP).scale(.5)
        env_map_text = MathTex('Env\ Map').scale(.5).next_to(env_map, UP)

        '''
        Loss Box
        '''
        loss_box_arrow = Arrow(DR.get_edge_center(RIGHT), DR.get_edge_center(RIGHT) + [1, 0, 0])
        loss_box_arrow_reverse = Arrow(*loss_box_arrow.get_start_and_end()[::-1])
        loss_box = Rectangle().next_to(loss_box_arrow.get_end(), RIGHT)
        loss_text = MathTex('\\bf{Loss}').scale(.65).set_color(RED)
        loss_text.next_to(loss_box, 2*DOWN)
        gt_text = MathTex('GT').scale(.5).next_to(loss_box, DOWN).shift(RIGHT)
        forward_text = MathTex('Forward').scale(.5).next_to(loss_box, DOWN).shift(LEFT)
        
        self.play(
                    FadeIn(DR),
                    FadeIn(dr_name),

                    # FadeIn(optimizing_texture),

                    FadeIn(proxy_geometery),
                    FadeIn(camera_params),

                    # FadeIn(env_map),
                    # FadeIn(loss_box),
                    
                    FadeIn(optimizing_texture_text),
                    FadeIn(env_map_text),
                    FadeIn(loss_text),
                    FadeIn(gt_text),
                    FadeIn(forward_text)
                 )
        self.play(
                    FadeIn(proxy_geometry_arrow),
                    FadeIn(camera_params_arrow),
                    FadeIn(env_map_arrow)
                 )
        self.play(
                    FadeIn(diffuse_input_arrow),
                    FadeIn(loss_box_arrow),
                 )

        self.play(
                    ReplacementTransform(loss_box_arrow.copy(), loss_box_arrow_reverse.set_color(BLUE)),
                    ReplacementTransform(diffuse_input_arrow.copy(), diffuse_optimized_arrow.set_color(BLUE))
                 )

        self.remove(loss_box_arrow_reverse)
        self.remove(diffuse_optimized_arrow)

        self.play(
                    ReplacementTransform(loss_box_arrow_reverse, loss_box_arrow.set_color(GREEN)),
                    ReplacementTransform(diffuse_optimized_arrow, diffuse_input_arrow.set_color(GREEN))
                 )
        self.remove(loss_box_arrow_reverse)
        self.remove(diffuse_optimized_arrow)


        '''
        Optimized Diffuse Texture
        '''
        optimized_material = ImageMobject('./resources/diffuse_opt.png').move_to(optimizing_texture.get_center())
        optimized_texture_text = MathTex('\\bf{Diffuse Color}').move_to(optimized_material.get_edge_center(LEFT) + [0, -.3, 0]).scale(.5).set_color(MAROON_A).rotate(TAU/4)

        bump_map = ImageMobject('./resources/bump_map.png').move_to([-9, .5, 0])
        bump_map_text = MathTex('\\bf{Bump Map}').move_to(bump_map.get_edge_center(LEFT) + [0, -.3, 0]).scale(.5).rotate(TAU/4)

        #remove instead of play and fadeout

        self.play(
                    # FadeIn(optimized_material),

                    # FadeOut(optimizing_texture),

                    FadeOut(DR),
                    FadeOut(dr_name),

                    FadeOut(proxy_geometery),
                    FadeOut(proxy_geometry_arrow),

                    FadeOut(camera_params),
                    FadeOut(camera_params_arrow),

                    # FadeOut(env_map),
                    FadeOut(env_map_arrow),

                    FadeOut(diffuse_optimized_arrow),
                    FadeOut(diffuse_input_arrow),
                    FadeOut(optimizing_texture_text),
                    
                    FadeOut(loss_box_arrow_reverse),
                    FadeOut(loss_box_arrow),

                    # FadeOut(loss_box),
                    FadeOut(env_map_text),
                    FadeOut(loss_text),
                    FadeOut(gt_text),
                    FadeOut(forward_text)
                )

        self.play(
                        ApplyMethod(bump_map.move_to, [-5.7, -1.8, 1]),
                        ApplyMethod(bump_map_text.move_to, [-6.9, -2.2, 1]),

                        ApplyMethod(optimized_material.shift, 2.5*DOWN + .5* LEFT),
                        ApplyMethod(optimized_texture_text.shift, 2.5*DOWN + .7* LEFT),
                        

                )

        '''
        BRDF CONSTRUCTION AND SAMPLING
        '''
        brdf_construction_n_sampling_rectangle = RoundedRectangle(height=4,width=2, color=PURPLE).scale(.6).move_to(optimized_material.get_edge_center(RIGHT) + [2, 0, 0])
        brdf_construction_n_sampling = MathTex('Sampling, Projection\\\\ and BRDF Construction').scale(.4).rotate(TAU/4).move_to(
                                                                                                                                brdf_construction_n_sampling_rectangle.get_center()
                                                                                                                            )

        brdf_extraction_arrow = Arrow(optimized_material.get_edge_center(RIGHT), brdf_construction_n_sampling_rectangle.get_edge_center(LEFT)).scale(.75).set_color(RED)

        '''
        BRDF Function
        '''
        brdf_function_rectangle = RoundedRectangle(fill_opacity=.3, color=BLUE_C).scale(.4).move_to(
                                                                                                        brdf_construction_n_sampling_rectangle.get_edge_center(RIGHT) + [2, 0, 0]
                                                                                                    )
        brdf_function = MathTex('BRDF\\\\Function').scale(.4).move_to(brdf_function_rectangle.get_center())

        brdf_function_arrow_left = Arrow(brdf_construction_n_sampling_rectangle.get_edge_center(RIGHT), brdf_function_rectangle.get_edge_center(LEFT)).scale(.75).set_color(RED)


        '''
        SH Projection
        '''
        sh_projection_rectangle = RoundedRectangle(height=4, width=2, color=GREEN).scale(.4).move_to(brdf_function_rectangle.get_edge_center(RIGHT) + [2, 0, 0])
        sh_projection = MathTex('SH Projection').scale(.4).rotate(TAU/4).move_to(sh_projection_rectangle.get_center())
        
        brdf_function_arrow_right = Arrow(brdf_function_rectangle.get_edge_center(RIGHT), sh_projection_rectangle.get_edge_center(LEFT)).scale(.75).set_color(RED)

        '''
        BRDF IN SH SPACE
        '''
        brdf_in_sh_space = Rectangle().scale(.4).move_to(sh_projection_rectangle.get_edge_center(RIGHT) + [2, 0, 0])
        brdf_sh_img = ImageMobject('./resources/brdf_sh.png').move_to(brdf_in_sh_space.get_center())

        ####################################
        #Change fr to bold face
        ####################################
        brdf_in_sh_space_text = MathTex('\\bf{\\overrightarrow{f_r}}',
                                        '\  (BRDF\ in\ SH\ Space)').scale(.4).move_to(brdf_sh_img.get_edge_center(DOWN) + [.25, -.125, 0])
        brdf_in_sh_projection_arrow = Arrow(sh_projection_rectangle.get_edge_center(RIGHT), brdf_in_sh_space.get_edge_center(LEFT)).scale(.75).set_color(RED)
        brdf_in_sh_space_text[0].set_color_by_gradient(MAROON_A, GREEN)
        brdf_in_sh_space_text[1].set_color_by_gradient(MAROON_A, GREEN)

        '''
        UV map
        '''
        uv_map_rectangle = Square().to_edge(UP+LEFT).scale(.5)
        uv_map_img = ImageMobject('./resources/uv_map.png').move_to(uv_map_rectangle.get_center())
        uv_map_text = MathTex('UV\ Map').next_to(uv_map_img, UP).scale(.5).set_color_by_gradient(BLUE, TEAL, LIGHT_PINK, PURPLE)
        uv_map_text.shift([0, -.3, 0])

        uv_map_arrow_1 = Line(uv_map_rectangle.get_edge_center(RIGHT) + [.5, 0, 0], uv_map_rectangle.get_edge_center(RIGHT) + [1, 0, 0]).set_color(RED)
        uv_map_arrow_2 = Line(uv_map_arrow_1.get_end(), uv_map_arrow_1.get_end() + [0, -4.5, 0]).set_color(RED)
        uv_map_arrow_3 = Line(uv_map_arrow_2.get_end(), uv_map_arrow_2.get_end() + [.5, 0, 0]).set_color(RED)
        uv_arrow_tip = ArrowTriangleFilledTip().rotate(-TAU/5.8).move_to(uv_map_arrow_3.get_end()).scale(.6).set_color(RED)


        '''
        Neural Texture
        '''
        neural_texture_rectangle = Square().next_to(uv_map_rectangle, 2*DOWN).scale(.5)
        neural_texture_img = ImageMobject('./resources/neural_texture.png').move_to(neural_texture_rectangle.get_center())
        neural_texture_text = MathTex('Neural\ Texture').next_to(neural_texture_img, DOWN).set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE).scale(.5)
        neural_texture_text.shift([0, .25, 0])

        '''
        Sampling and Projection
        '''
        sampling_projection_rectangle = RoundedRectangle(height=4,width=2, color=PURPLE).scale(.6).move_to(
                                                                                                            (neural_texture_rectangle.get_edge_center(RIGHT) + uv_map_rectangle.get_edge_center(RIGHT))/2+[2, 0, 0]
                                                                                                        )

        sampling_projection = MathTex('Sampling\ and\ Projection').scale(.4).rotate(TAU/4).move_to(
                                                                                                sampling_projection_rectangle.get_center()
                                                                                            )
        sampling_arrow_from_uv = Arrow(uv_map_rectangle.get_edge_center(RIGHT), sampling_projection_rectangle.get_edge_center(LEFT)).set_color(BLUE)
        sampling_arrow_from_neural_texture = Arrow(neural_texture_rectangle.get_edge_center(RIGHT), sampling_projection_rectangle.get_edge_center(LEFT)).set_color(BLUE)

        '''
        Sampled Data
        '''
        sample_data_rectangle = Square().scale(.5).move_to(sampling_projection_rectangle.get_edge_center(RIGHT) + [1.5, 0, 0])
        sample_data_img = ImageMobject('./resources/sampled.png').move_to(sample_data_rectangle.get_center())
        sampled_data_arrow = Arrow(sampling_projection_rectangle.get_edge_center(RIGHT), sample_data_rectangle.get_edge_center(LEFT)).set_color(BLUE)

        '''
        Neural Renderer
        '''
        neural_renderer = [
                            Rectangle(height=4, width=.5).scale(.5).move_to(sample_data_rectangle.get_edge_center(RIGHT) + [1, 0, 0]),
                            Rectangle(height=3, width=.5).scale(.5).move_to(sample_data_rectangle.get_edge_center(RIGHT) + [1.5, 0, 0]),
                            Rectangle(height=3, width=.5).scale(.5).move_to(sample_data_rectangle.get_edge_center(RIGHT) + [2, 0, 0]),
                            Rectangle(height=4, width=.5).scale(.5).move_to(sample_data_rectangle.get_edge_center(RIGHT) + [2.5, 0, 0]),
                          ]
        neural_renderer_text = MathTex('Neural\ Renderer').scale(.5)
        neural_renderer_text.move_to(
                                        (neural_renderer[0].get_edge_center(DOWN) + neural_renderer[-1].get_edge_center(DOWN)) / 2 + [0, -.25, 0]
                                    )

        neural_renderer_arrow = Arrow(sample_data_rectangle.get_edge_center(RIGHT), neural_renderer[0].get_edge_center(LEFT)).set_color(BLUE)


        '''
        LIF SH Projection
        '''
        lif_sh_projection_rectangle = Rectangle().scale(.5)
        lif_sh_projection_rectangle.move_to(neural_renderer[-1].get_edge_center(RIGHT) + [2., 1, 0])
        lif_img = ImageMobject('./resources/lif.png').move_to(lif_sh_projection_rectangle.get_center())
        lif_sh_projection = MathTex('\\bf{\\overrightarrow {T}}',
                                '\ (LIF\ in\ SH\ Space)').scale(0.5).move_to(lif_img.get_edge_center(UP) + [1., 0.125, 0])
        lif_sh_projection[0].set_color_by_gradient(ORANGE, GREEN)
        lif_sh_projection[1].set_color_by_gradient(ORANGE, GREEN)

        lif_arrow = Arrow(neural_renderer[-1].get_edge_center(RIGHT), lif_sh_projection_rectangle.get_edge_center(LEFT)).scale(.5).set_color(BLUE)
        

        '''
        Final Render
        '''
        final_render_rectangle = Rectangle().scale(.5)

        final_render_rectangle.to_edge(RIGHT)
        final_render_rectangle.shift([0, .5, 0])

        final_render_img = ImageMobject('./resources/rendered.png').move_to(final_render_rectangle.get_center())
        final_render_text = MathTex('\\bf{\\overrightarrow {f_r}',
                                     '\cdot',
                                     ' \\overrightarrow {T}}\\\\',
                                    'Rendered').scale(.5).rotate(TAU/4)

        final_render_text.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)

        final_render_text[0].set_color_by_gradient(MAROON_A, GREEN)
        final_render_text[2].set_color_by_gradient(ORANGE, GREEN)

        final_render_text.move_to(final_render_img.get_edge_center(LEFT) - [0.4, 0., 0])
        final_render_from_lif_arrow = Arrow(lif_sh_projection_rectangle.get_edge_center(DOWN), final_render_rectangle.get_edge_center(UP)).set_color_by_gradient(ORANGE, GREEN)

        final_render_from_brdf_arrow = Arrow(brdf_in_sh_space.get_edge_center(UP), final_render_rectangle.get_edge_center(DOWN)).set_color_by_gradient(MAROON_A, GREEN)

      
        '''
        Learnable
        '''

        learnable = RoundedRectangle(height=4.5, width=14, color=PINK).shift(1.5 * UP)
        learnable_text = MathTex('\\bf{ Learnable }').scale(.65).set_color(PINK)
        learnable_text.move_to(learnable.get_edge_center(DOWN) + [0, .25, 0])

        '''
        Our Method
        '''
        title = MathTex('\\bf{Our Method}').scale(.5)
        title.move_to(learnable.get_edge_center(UP) + [0, -.25, 0])

        '''
        Render
        '''
        self.play(
                    FadeIn(uv_map_text),  
                    FadeIn(neural_texture_text),
                    FadeIn(neural_renderer_text),
                    FadeIn(lif_sh_projection),

                    FadeIn(brdf_construction_n_sampling),
                    FadeIn(brdf_construction_n_sampling_rectangle),

                    FadeIn(brdf_function),
                    FadeIn(brdf_function_rectangle),

                    FadeIn(brdf_sh_img),
                    FadeIn(sh_projection),
                    FadeIn(sh_projection_rectangle),

                    # FadeIn(brdf_in_sh_space),
                    # # FadeIn(uv_map_rectangle),
                    # # FadeIn(neural_texture_rectangle),
                    # # FadeIn(sample_data_rectangle),
                    # FadeIn(lif_sh_projection_rectangle),
                    # FadeIn(final_render_rectangle),

                    FadeIn(uv_map_img),
                    FadeIn(neural_texture_img),
                    FadeIn(sample_data_img),
                    FadeIn(lif_img),
                    FadeIn(final_render_img),

                    *[FadeIn(neural_layer) for neural_layer in neural_renderer],

                    FadeIn(sampling_projection_rectangle),
                    FadeIn(sampling_projection),
                 )

        self.play(
                    FadeIn(uv_map_arrow_1),
                    FadeIn(uv_map_arrow_2),
                    FadeIn(uv_map_arrow_3),
                    FadeIn(uv_arrow_tip),

                    FadeIn(brdf_extraction_arrow),
                    FadeIn(brdf_function_arrow_left),
                    FadeIn(brdf_function_arrow_right),

                    FadeIn(brdf_in_sh_space_text),
                    FadeIn(brdf_in_sh_projection_arrow),

                    FadeIn(sampling_arrow_from_uv),
                    FadeIn(sampling_arrow_from_neural_texture),

                    FadeIn(sampled_data_arrow),

                    FadeIn(neural_renderer_arrow),

                    FadeIn(lif_arrow),

                    FadeIn(final_render_text),

                    ShowCreation(final_render_from_lif_arrow),
                    ShowCreation(final_render_from_brdf_arrow),
                    
                    FadeIn(learnable),
                    Write(learnable_text),

                    Write(title)
                )
        
        

        self.wait(2)
        
                                        
