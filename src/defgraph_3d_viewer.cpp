#include "defgraph_3d_viewer.h"

#include <pangolin/pangolin.h>
#include <Eigen/Dense>
#include <iostream>
#include <pangolin/var/varextra.h>  // Pushed



// ---- Dibujar pirámide de cámara ----
static void DrawCameraFrustum(const Eigen::Matrix3d& Rwc,
                              const Eigen::Vector3d& pwc,
                              float scale = 0.25f)
{
    // Coordenadas en marco de cámara C
    Eigen::Vector3d C(0,0,0);
    Eigen::Vector3d p1(-0.2, -0.15, 0.5);
    Eigen::Vector3d p2( 0.2, -0.15, 0.5);
    Eigen::Vector3d p3( 0.2,  0.15, 0.5);
    Eigen::Vector3d p4(-0.2,  0.15, 0.5);

    C  *= scale;
    p1 *= scale;
    p2 *= scale;
    p3 *= scale;
    p4 *= scale;

    auto toW = [&](const Eigen::Vector3d& Xc){
        return Rwc * Xc + pwc;
    };

    Eigen::Vector3d Cw  = toW(C);
    Eigen::Vector3d p1w = toW(p1);
    Eigen::Vector3d p2w = toW(p2);
    Eigen::Vector3d p3w = toW(p3);
    Eigen::Vector3d p4w = toW(p4);

    glLineWidth(1.5f);
    glColor3f(1.0, 1.0, 0.0); // amarillo

    glBegin(GL_LINES);
    // laterales
    glVertex3dv(Cw.data()); glVertex3dv(p1w.data());
    glVertex3dv(Cw.data()); glVertex3dv(p2w.data());
    glVertex3dv(Cw.data()); glVertex3dv(p3w.data());
    glVertex3dv(Cw.data()); glVertex3dv(p4w.data());

    // plano imagen
    glVertex3dv(p1w.data()); glVertex3dv(p2w.data());
    glVertex3dv(p2w.data()); glVertex3dv(p3w.data());
    glVertex3dv(p3w.data()); glVertex3dv(p4w.data());
    glVertex3dv(p4w.data()); glVertex3dv(p1w.data());
    glEnd();
}


// ---- VIEWER PRINCIPAL ----
void runDeformationGraph3D(
    const std::unordered_map<NodeId, NodeRest>& nodes_rest,
    const std::map<int, NodePositions>& nodes_per_kf,
    const std::vector<int>& optimized_kfs,
    const std::unordered_map<int, KFPose>& cam_poses,
    const std::vector<ViscousEdge>& viscous_edges)
{
    if (optimized_kfs.empty()) {
        std::cerr << "[DEF3D] No hay KFs optimizados.\n";
        return;
    }

    // ---- Inicializar Pangolin ----
    pangolin::CreateWindowAndBind("DefGraph3D", 1280, 720);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1280,720,700,700,640,360,0.01,200),
        pangolin::ModelViewLookAt(0, -2, -4, 0,0,0, 0,-1,0)
    );

    pangolin::View& d_cam =
        pangolin::CreateDisplay()
        .SetBounds(0,1, pangolin::Attach::Pix(250),1, -1280.0/720.0)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::CreatePanel("ui").SetBounds(0,1,0,pangolin::Attach::Pix(250));

    pangolin::Var<int>  ui_kf("ui.KF", 0, 0, (int)optimized_kfs.size()-1);
    pangolin::Var<bool> ui_next("ui.Next", false, false);
    pangolin::Var<bool> ui_prev("ui.Prev", false, false);
    pangolin::Var<bool> ui_play("ui.Play", false, false);

    std::cout << "[DEF3D] Navega KFs con slider, botones Next/Prev o Play.\n";

    static int frame_counter = 0;

    // ---- Loop principal ----
    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // botones Next / Prev
        if (pangolin::Pushed(ui_next)) {
            ui_kf = std::min((int)ui_kf + 1, (int)optimized_kfs.size()-1);
        }
        if (pangolin::Pushed(ui_prev)) {
            ui_kf = std::max((int)ui_kf - 1, 0);
        }

        // modo Play: avanza automáticamente cada N frames
        if (ui_play) {
            frame_counter++;
            if (frame_counter % 5 == 0) { // cambia el 5 para más rápido/lento
                int next_idx = (int)ui_kf + 1;
                if (next_idx >= (int)optimized_kfs.size()) next_idx = 0;
                ui_kf = next_idx;
            }
        }

        int idx = std::max(0, std::min((int)ui_kf, (int)optimized_kfs.size()-1));
        int k = optimized_kfs[idx];

        // nodos deformados del KF actual
        const auto& nodes_curr = nodes_per_kf.at(k);

        // pose de cámara del KF
        Eigen::Matrix3d Rwc = cam_poses.at(k).R;
        Eigen::Vector3d pwc = cam_poses.at(k).p;

        // ===== DIBUJO =====

        // ejes
        glLineWidth(2);
        glBegin(GL_LINES);
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(0.5,0,0);
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,0.5,0);
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,0.5);
        glEnd();

        // nodos en reposo
        glPointSize(3);
        glBegin(GL_POINTS);
        glColor3f(0.4f,0.4f,0.4f);
        for (auto& kv: nodes_rest) {
            glVertex3dv(kv.second.X_ref.data());
        }
        glEnd();

        // aristas en reposo
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 1.0f); // magenta
        for (const auto& e : viscous_edges) {
            auto it_i = nodes_rest.find(e.i);
            auto it_j = nodes_rest.find(e.j);
            if (it_i == nodes_rest.end() || it_j == nodes_rest.end()) continue;

            glVertex3dv(it_i->second.X_ref.data());
            glVertex3dv(it_j->second.X_ref.data());
        }
        glEnd();

        // nodos deformados
        glPointSize(5);
        glBegin(GL_POINTS);
        glColor3f(0.0f,1.0f,0.0f);
        for (auto& kv: nodes_curr) {
            glVertex3dv(kv.second.data());
        }
        glEnd();

        // aristas deformadas
        glLineWidth(1.5);
        glBegin(GL_LINES);
        glColor3f(0.0f,0.0f,1.0f);
        for (auto& e : viscous_edges) {
            if (!nodes_curr.count(e.i) || !nodes_curr.count(e.j)) continue;
            glVertex3dv(nodes_curr.at(e.i).data());
            glVertex3dv(nodes_curr.at(e.j).data());
        }
        glEnd();

        // trayectoria de cámara (todas las poses)
        glLineWidth(1.2f);
        glBegin(GL_LINE_STRIP);
        glColor3f(1.0f, 0.8f, 0.0f); // naranja
        for (int kk : optimized_kfs) {
            auto itp = cam_poses.find(kk);
            if (itp == cam_poses.end()) continue;
            glVertex3dv(itp->second.p.data());
        }
        glEnd();

        // pirámide de cámara del KF actual
        DrawCameraFrustum(Rwc, pwc);

        pangolin::FinishFrame();
    }
}
