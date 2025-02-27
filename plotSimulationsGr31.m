

for i=1:N
    x_noisy(:,i) = Q{i};
    x_clean(:,i) = Q_true{i};
end 
figure('DefaultAxesFontSize',25,'DefaultAxesFontWeight','bold');
[l1,l2,l3]=sphere(20); 
surf(l1,l2,l3,'FaceColor','w','EdgeColor', [.7 .7 .7], 'FaceAlpha', 0);  % plot the sphere with green color
hold on;
plot3(p0_true(1),p0_true(2),p0_true(3),'m*','Linewidth',2)
hold on;
plot3(p1_true(1),p1_true(2),p1_true(3),'m*','Linewidth',2)
hold on;
plot3(v0_true(1)+p0_true(1),v0_true(2)+p0_true(2),v0_true(3)+p0_true(3),'m*','Linewidth',2)
hold on;
plot3([p0_true(1),v0_true(1)+p0_true(1)],[p0_true(2),v0_true(2)+p0_true(2)],[p0_true(3),v0_true(3)+p0_true(3)],'g','Linewidth',3)
hold on;
plot3(x_clean(1,:) , x_clean(2,:), x_clean(3,:), '-b','Linewidth',3)
text(p0_true(1),p0_true(2),p0_true(3),'$\gamma(0)$','Interpreter','latex')
text(p1_true(1),p1_true(2),p1_true(3),'$\gamma(1)$','Interpreter','latex')
text(v0_true(1)+p0_true(1),v0_true(2)+p0_true(2),v0_true(3)+p0_true(3),'$\dot{\gamma}(0)$','Interpreter','latex')
axis equal
grid off
axis off
xlim([-1.2, 1.2])
ylim([-1.2, 1.2])
zlim([-1.2, 1.2])
%title('Initial conditions on Gr(3,1)')
view(-120,-50)
set(gcf,'color','w'); % Set background color to white
set(gca,'box','off'); % Turn off the axes box
%saveas(gcf,'initial_Gr31.png');


figure('DefaultAxesFontSize',25,'DefaultAxesFontWeight','bold');
[l1,l2,l3]=sphere(20); 
surf(l1,l2,l3,'FaceColor','w','EdgeColor', [.7 .7 .7], 'FaceAlpha', 0);  % plot the sphere with green color
hold on;
plot3(x_noisy(1,:) , x_noisy(2,:), x_noisy(3,:), 'ks','Linewidth',2)
hold on;
plot3(x_clean(1,:) , x_clean(2,:), x_clean(3,:), '-b','Linewidth',3)
axis equal
grid off
axis off
xlim([-1.2, 1.2])
ylim([-1.2, 1.2])
zlim([-1.2, 1.2])
%title('Simulations on Gr(3,1)')
view(-120,-50)
set(gcf,'color','w'); % Set background color to white
set(gca,'box','off'); % Turn off the axes box
%saveas(gcf,'noise_Gr31.png');

