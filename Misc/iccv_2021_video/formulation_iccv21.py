from manim import *


wait_time = .5
class Formulation(Scene):
    def construct(self):

        '''
            Title
        '''
        title = MathTex('Appearance\ Editing', '\ with\ Free-viewpoint\ Neural\ Rendering')

        paper_id = MathTex('Paper\ ID:\ 7013').set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE).next_to(title, DOWN)
        title[0].set_color(ORANGE)
        title[1].set_color(WHITE)
        
        title.to_edge(UP+DOWN)

        self.play(
                    FadeIn(title),
                    FadeIn(paper_id)
                 )
        self.wait(2)

        render_eq_title = MathTex('The\ Rendering\ Equation').set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)

        self.play(
                    ReplacementTransform(title, render_eq_title),
                    ReplacementTransform(paper_id, render_eq_title)
                    )

        '''
            Rendering Eq
        '''
        render_eq = MathTex("L(x)",                 #0

                            ' = ',                  #1

                            '\int_',                #2
                            '{\Omega}',             #3

                            'f_r(',                 #4
                            'p_x',                  #5
                            ', ',                   #6
                            '\omega_x',             #7
                            ', ',                   #8
                            '\omega_i',             #9
                            ')',                    #10

                            'L_e(',                 #11
                            '\omega_i',             #12
                            ')',                    #13

                            'V(',                   #14
                            'p_x',                  #15
                            ', ',                   #16
                            '\omega_i',             #17
                            ')',                    #18

                            '(',                    #19
                            '\omega_i',             #20
                            '\cdot',                #21
                            'n',                    #22
                            ')',                    #23

                            'd\omega_i',            #24
                            
                            '=',                    #25
                            
                            '\int_{\Omega}',        #26
                            
                            'f_r(',                 #27
                            'p_x',                  #28
                            ', ',                   #29
                            '\omega_x',             #30
                            ', ',                   #31
                            '\omega_i',             #32
                            ')',                    #33
                            
                            'T(',                   #34
                            'p_x',                  #35
                            ', ',                   #36
                            '\omega_i',             #37
                            ')',                    #38
                            
                            'd\omega_i'             #39
                            ) 
        render_eq.scale(.8)

        self.play(
                    ApplyMethod(render_eq_title.shift, 3 * UP),
                    FadeIn(render_eq[:25])
                 )

        render_eq_parts = [
                                render_eq[2:4], #Integral\ Over\ hemisphere\ \Omega
                                render_eq[4:11], #BRDF
                                render_eq[11: 14], #Illumination
                                render_eq[14: 19], #Visibility Function
                                render_eq[19: 24], #Cosine Foreshortenting Factor
                                render_eq[0], #Incoming\ Radiance\ L\ at\ pixel\ x\ as\ seen\ from\ the\ direction\ of\ \omega_x
                          ]
        render_eq_colors = [ 
                            TEAL_C,
                            MAROON_A,
                            RED,
                            YELLOW,
                            BLUE,
                            GOLD,
                           ]

        render_eq[26].set_color(TEAL_C)
        render_eq[27: 34].set_color(MAROON_A)
        render_eq[34: 39].set_color(ORANGE)

        about = MathTex(
                            '\\bf{Integral\ Over\ hemisphere\ \Omega',
                            'BRDF',
                            'Illumination',
                            'Visibility\ Function',
                            'Cosine\ Fore\-shortening\ Factor',
                            'Incoming\ Radiance\ L\ at\ pixel\ x\ as\ seen\ from\ the\ direction\ of\ \omega_x',

                            '3D\ point\ corresponding\ to\ the\ 2D\ pixel\ x',
                            'Direction\ from\ p_x\ towards\ Camera',
                            'Incoming\ light\ direction',
                            'Local\ Irradiance\ Function\ (LIF)}',
                        ).scale(.5)

        about_counter = 0

        render_eq_animations = []
        for i in range(len(render_eq_parts)):

            render_eq_animations += [
                                                Write(render_eq_parts[i].set_color(render_eq_colors[i])) , 
                                                FadeIn(about[about_counter].set_color(render_eq_colors[i]).next_to(render_eq, (i+1) * DOWN))
                                    ]
            about_counter += 1
                                
        self.play(*render_eq_animations)
        self.wait(1)
        self.remove(*about)

        #Sub-parts
        p_x = [render_eq[5], render_eq[15]]
        p_x_box = [SurroundingRectangle(highlight, buff=.05) for highlight in p_x]
        
        omega_x = [render_eq[7]]
        omega_x_box = [SurroundingRectangle(highlight, buff=.05) for highlight in omega_x]

        omega_i = [render_eq[9], render_eq[12], render_eq[17], render_eq[20]]
        omega_i_box = [SurroundingRectangle(highlight, buff=.05) for highlight in omega_i]

        highlights_boxes = [p_x_box, omega_x_box, omega_i_box]

        self.remove(*about)
        subpart_animations = []
        subpart_color = [PURPLE_C, TEAL_B, YELLOW]
        for i in range(len(highlights_boxes)):
            subpart_animations += [ShowCreation(box.set_color(subpart_color[i])) for box in highlights_boxes[i]]
            subpart_animations += [FadeIn(about[about_counter].set_color(subpart_color[i]).next_to(render_eq, (i +1)*DOWN))]
            about_counter+=1
        
        self.play(*subpart_animations)
        self.wait(1)

        uncreate_boxes = [Uncreate(box) for i in range(len(highlights_boxes)) for box in highlights_boxes[i] ]
        self.remove(*about)
        
        self.play(
                    *uncreate_boxes,
                    Write(render_eq[25])
                 )

        self.play(
                    ReplacementTransform(render_eq[2: 4].copy(), render_eq[26])
                 )

        self.wait(wait_time)
        self.play(
                    ReplacementTransform(render_eq[4: 11].copy(), render_eq[27: 34])
                )

        self.wait(wait_time)
        self.play(
                    ReplacementTransform(render_eq[11: 24].copy(), render_eq[34: 39]),
                    ApplyMethod(render_eq[11: 24].set_color, ORANGE)
                 )
        
        self.wait(wait_time)
        self.play(
                    ReplacementTransform(render_eq[24].copy(), render_eq[-1])
                 )

        self.wait(wait_time)
        self.play( 
                    Uncreate(render_eq_title),
                    ApplyMethod(render_eq.shift, 2 * UP)
                 )

        LIF = [render_eq[34: 39]]
        highlights_boxes =[SurroundingRectangle(highlight, buff=.01) for highlight in LIF]

        for i in range(len(highlights_boxes)):
            self.play(
                            *[ShowCreation(box) for box in highlights_boxes[i]],
                            FadeIn(about[about_counter].next_to(render_eq, DOWN))
                      )
            self.wait(wait_time * 2)
            [self.remove(box) for box in highlights_boxes[i]]

            about_counter+=1

        self.remove(*about)

        ideology = MathTex(
                            r'The\ above\ equation\ for\ L(x)\ can\ be\ evaluated\ by\ projecting\ the\ ',
                            r'LIF',
                            r'\ and\ ',
                            r'BRDF',
                            r'\ to\ ',
                            r'Spheircal\ Harmonics(SH)\ basis'
                       ).scale(.5)

        ideology.set_color(WHITE)
        ideology[1].set_color(ORANGE)
        ideology[3].set_color(MAROON_A)
        ideology[-1].set_color(GREEN)

        new_rendering_eq = MathTex(
                                    "L(x)",                 #0

                                    ' = ',                  #1
                                    
                                    '\int_{\Omega}',        #2
                                    
                                    'f_r(',                 #3
                                    'p_x',                  #4
                                    ', ',                   #5
                                    '\omega_x',             #6
                                    ', ',                   #7
                                    '\omega_i',             #8
                                    ')',                    #9
                                    
                                    'T(',                   #10
                                    'p_x',                  #11
                                    ', ',                   #12
                                    '\omega_i',             #13
                                    ')',                    #14
                                    
                                    'd\omega_i'             #15
                                  )

        new_rendering_eq.move_to(2*LEFT)
        new_rendering_eq.set_color(WHITE)
        new_rendering_eq[0].set_color(GOLD)
        new_rendering_eq[2].set_color(TEAL_C)
        new_rendering_eq[3:10].set_color(MAROON_A)
        new_rendering_eq[10:14].set_color(ORANGE)
        ideology.move_to(2*DOWN),
        self.play(
                    ReplacementTransform(render_eq, new_rendering_eq),
                    FadeIn(ideology)
                 )
        
        sh_projection = MathTex(
                                    '=',
                                    '\\overrightarrow f_r',
                                    '\cdot',
                                    '\\overrightarrow T'
                                )
        
        sh_projection.next_to(new_rendering_eq, RIGHT)
        sh_projection[1].set_color_by_gradient(MAROON_A,GREEN)
        sh_projection[-1].set_color_by_gradient(ORANGE,GREEN)
        brace1 = Brace(new_rendering_eq[3: 10], DOWN, buff=SMALL_BUFF)
        brace2 = Brace(new_rendering_eq[10: 15], DOWN, buff=SMALL_BUFF)

        text1 = brace1.get_tex('BRDF\ projection\ to\ SH').set_color_by_gradient(MAROON_A,GREEN)
        text2 = brace2.get_tex('LIF\ projection\ to\ SH').set_color_by_gradient(ORANGE,GREEN)


        self.play(
                    GrowFromCenter(brace1),
                    FadeIn(text1),
                    ReplacementTransform(new_rendering_eq[3:10].copy(), sh_projection[1]),
                 )
        self.wait(wait_time)
        self.play(
                    ReplacementTransform(brace1, brace2),
                    ReplacementTransform(new_rendering_eq[10: 14].copy(), sh_projection[3]),
                    ReplacementTransform(text1 , text2),
                 )
        self.wait(wait_time)
        self.play(
                    Uncreate(text2), 
                    Uncreate(brace2)
                )
        
        sh_projection_ideology = MathTex(
                                            'The\ Integral\ for\ evaluation\ of\ L(x)\ is\ dot\ product\ of\ ',
                                            'BRDF',
                                            '\ and\ ',
                                            'LIF\ ',
                                            'projected\ individually\ to\ the\ SH\ basis'
                                        ).scale(.5)

        sh_projection_ideology.set_color(WHITE)
        sh_projection_ideology.move_to(2 * DOWN)
        sh_projection_ideology[1].set_color_by_gradient(MAROON_A, GREEN)
        sh_projection_ideology[-2].set_color_by_gradient(ORANGE, GREEN)

        self.play(
                    ReplacementTransform(ideology, sh_projection_ideology),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                    FadeIn(sh_projection[2]),
                 )

        self.play(FadeIn(sh_projection[0]))
        self.wait(wait_time)