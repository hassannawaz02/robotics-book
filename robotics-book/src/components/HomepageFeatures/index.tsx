import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'ROS 2',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Master ROS 2 framework for robotic applications and communication protocols.
      </>
    ),
  },
  {
    title: 'Simulation',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Explore Gazebo and Unity for testing robotics algorithms in virtual worlds.
      </>
    ),
  },
  {
    title: 'NVIDIA Isaac™',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Develop AI-powered robotics with GPU acceleration for vision and control.
      </>
    ),
  },
  {
    title: 'VLA (Vision-Language-Action)',
    Svg: require('@site/static/img/undraw_vla.svg').default,
    description: (
      <>
        Build multimodal AI systems integrating vision, language, and robotic action.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <div className={styles.featureCard}>
        <div className="text--center">
          <Svg className={styles.featureSvg} role="img" />
        </div>
        <div className="text--center padding-horiz--md">
          <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
          <p className={styles.featureDescription}>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center padding-bottom--lg">
          <Heading as="h2" className={styles.featureHeading}>Course Modules</Heading>
          <p className={styles.featureSubHeading}>Comprehensive curriculum designed for next-generation robotics developers</p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
